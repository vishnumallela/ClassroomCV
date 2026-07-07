# ML Pipeline Improvement Plan

Scope: services/ml-service (detector.py, merge.py, roles.py, events.py, jobs.py, db.py, config.py) plus the TimescaleDB schema owned by apps/api-service/drizzle. Ultralytics pinned at 8.4.89; all cited line numbers verified against the installed venv.

**Verified demo baseline** (68 min, 2560x1440 CCTV, ~30 uniformed students, 1 walking teacher): `teacher_present_ms=161800`, `teacher_board_ms=41200`, `entries=4`, `exits=4`, `max_students=30`, `avg_students=25.5`, 42 merged identities from ~31k detections, 156 raw BoT-SORT tracks. Every change below must be re-verified against these numbers, which is why the measurement harness (section 5) lands first.

**Reconciled facts** (where research lenses disagreed):

- The "minutes-long merge" claim is wrong today: `merge.merge_tracks` measured **0.12 s** on the real demo (31,300 detections, 156 raw tracks). The minutes live in the detection stage (progress 0..0.8). Merge scaling is still a real problem for live streams: per-pair `np.corrcoef` at 28.5 us/pair reaches ~32 s at n=1500 raw tracks, so scale-proofing stays on the roadmap but at lower priority.
- Three lenses proposed conflicting tracker yamls. Section 2 defines the single canonical `app/trackers/classroom_botsort.yaml`; sections 1 and 3 reference it instead of shipping their own.
- Detection `conf` kwarg: reconciled to `conf=0.05` paired with `track_low_thresh: 0.05` (the tracker-lens package is internally coherent: low-conf detections can extend tracks via BYTE stage 2 but can never create them because `new_track_thresh: 0.40`). The alternative `conf=0.15` would starve stage 2 of exactly the back-row band it exists for.

---

## 1. Detection (detector.py + config.py)

### Gaps

- `_track_frame` passes no `imgsz`, so ultralytics defaults to 640: the 2560x1440 frame is letterboxed to 640x384 (4x linear downscale). Back-row students (~80-120 px tall native) become 20-30 px; keypoints there are noise, so the hip/knee standing fallback and `back_to_camera` are effectively random exactly where occlusion is worst.
- `model.track()` silently floors detector conf at 0.1 (engine/model.py:581); no explicit `conf` or `max_det` is set, so behavior is implicit and version-fragile.
- No fp16 on MPS despite AutoBackend support and 2x fp16 throughput on the M3 Pro GPU.
- `KPT_CONF_LOW=0.3` gates the standing keypoint fallback with no minimum box-size requirement, so spatially meaningless keypoints on tiny boxes pass the gate and inject noise into the 0.30-weighted standing feature in roles.py.
- Stock `botsort.yaml` is hardcoded as a string in two places in `_track_frame` (lines 165 and 179); all tracker gaps are covered in section 2.

### Improvements

| ID | Change | Target | Rationale | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| D1 | Add `imgsz=1280` to both `model.track` calls; expose as `Settings.imgsz`. Try 1536 only if back-row recall still short after measuring. | detector.py:`_track_frame`, config.py:`Settings` | 640 quarters the linear resolution of a 2560-wide source; small-person detection and keypoint quality scale strongly with input size. The single largest quality lever in the pipeline. | Back-row recall up substantially; keypoints usable for standing/back_to_camera on far seats; better boxes mean better IoU association, fewer fragments into merge. Cost ~3x inference (40 ms to 120-150 ms/frame; 68-min video from ~15 to ~45 min, mostly recovered by D2). | low | Wall time triples without D2; conf distributions shift, re-verify baseline. |
| D2 | Add `half=True` to `model.track` only when effective device is mps; keep fp32 on the cpu fallback path. | detector.py:`_track_frame` | M3 Pro has 2x fp16 throughput; 1.3-1.6x end-to-end speedup offsets most of D1. | ~30-40% inference time cut at unchanged accuracy (conf drift typically < 0.01). | low | Minor conf drift; one baseline re-run to confirm. |
| D3 | Pass `conf=0.05` and `max_det=100` explicitly (pairs with `track_low_thresh: 0.05` in the section 2 yaml). Keep `iou=0.7` (lowering NMS IoU would suppress adjacent seated students). | detector.py:`_track_frame` | The silent 0.1 floor hides 0.05-0.1 detections (far back-row heads) from BYTE stage 2. With `fuse_score: false` and `new_track_thresh: 0.40`, these can only sustain existing tracks, never spawn noise tracks. `max_det=100` bounds NMS vs the 300 default with ~31 real people. | Back-row presence continuity improves; fewer spurious 5 s presence-gap events; deterministic behavior across ultralytics upgrades. | low | Slightly more boxes per association step (negligible); stored rows barely change since only ID-carrying boxes are emitted. |
| D4 | For the hip/knee standing fallback only: require kpt conf > 0.4 (new `STANDING_KPT_CONF`; leave `KPT_CONF_LOW=0.3` for `_back_to_camera`/`_torso_hist`) AND normalized box height >= 90/1440 before trusting keypoint geometry; below that return the aspect-only result. | detector.py:`_is_standing` | Far-seat keypoints pass the 0.3 gate while being spatially meaningless, polluting the highest-weighted role feature. | Stabler standing signal for the ~25 back/middle-row students; fewer students misclassified as teacher candidates. | low | Slightly fewer true standing flags for far students; acceptable since the teacher is large in frame. |
| D5 | Do NOT add SAHI/tiling; keep `sample_fps=5`. | detector.py:`detect_video` (decision, no change) | 2x2 tiles = 5-6 forward passes/frame (4x cost on 20,400 frames) and `model.track()` cannot consume externally merged sliced detections without reimplementing the BoT-SORT loop. At 2-3 fps a 1.5 m/s teacher displaces 0.5 m+ between samples and IoU association fails. D1 at 1280-1536 recovers most of the recall tiling would buy. | Avoids a 4x cost increase; preserves entry/exit fidelity. | none | If recall still short at 1536, revisit tiling as a detached second pass for occupancy counting only (no tracking). |
| D6 | Defer yolo11m-pose to yolo11l-pose swap; benchmark l at `imgsz=1280 half=True` on one video only after D1/D2 land. | config.py:`Settings.model_name` | For small objects, resolution buys far more AP than capacity (l is ~1.9x FLOPs for ~1-2 pose-AP); l at 1280 may exceed 250 ms/frame on MPS. | Correct ordering of compute spend. | low | None (deferral). |

### Quick wins

- Four kwargs in `_track_frame`: `imgsz=1280, half=True, conf=0.05, max_det=100`. The imgsz alone is the largest quality lever in the pipeline.
- Guard `half=True` behind `get_device()=='mps'` so the cpu fallback stays fp32.
- Expose `imgsz`, `conf`, `sample_fps`, `tracker_cfg` as `Settings` fields so tuning never requires code edits.

---

## 2. Tracking (canonical BoT-SORT config)

### Gaps

- `fuse_score: True` in 8.4.89 also applies inside `_second_association` with a fixed 0.5 gate (byte_tracker.py:423-425): cost = 1 - IoU*conf, and stage-2 detections have conf < 0.25 by construction, so minimum cost is 0.75 > 0.5. ByteTrack's low-conf recovery (the mechanism meant to keep back-row students tracked through conf dips) is **mathematically dead** today.
- `gmc_method: sparseOptFlow` runs goodFeaturesToTrack + pyramidal LK + RANSAC per sampled frame on a 1280x720 gray downscale for a bolted-down camera; pure CPU waste, and a foreground walking teacher can bias the "camera motion" warp, drifting all Kalman states.
- `track_buffer=30` is counted in SAMPLED frames with no fps scaling (byte_tracker.py:261 sets `max_frames_lost = track_buffer`), so 30 = 6 s at 5 fps: shorter than typical desk-occlusion plus head-down episodes.
- `new_track_thresh=0.25` lets one noisy low-conf frame spawn a raw track; every phantom becomes a merge candidate.
- `with_reid: False` while downstream re-ID leans on HSV torso histograms that uniforms make near-worthless.
- No fragmentation metric is logged, so tracker tuning cannot be validated (see E6).

### The canonical yaml

Create `app/trackers/classroom_botsort.yaml`, add `Settings.tracker_cfg` (absolute path), and pass `tracker=settings.tracker_cfg` in BOTH `model.track()` calls in `_track_frame` (`check_yaml` in trackers/track.py:47 accepts absolute paths; `persist=True` means the yaml is read once per process, restart the service after edits).

```yaml
tracker_type: botsort
track_high_thresh: 0.25   # stock; raise to 0.35 only if phantom tracks persist after new_track_thresh change
track_low_thresh: 0.05    # pairs with conf=0.05 in D3
new_track_thresh: 0.40    # spawning gate only; low-conf flicker still extends tracks via stage 1/2 (byte_tracker.py:474)
track_buffer: 60          # SAMPLED frames, no fps scaling in 8.4.89 -> 12 s at 5 fps
match_thresh: 0.8
fuse_score: false         # resurrects stage-2 low-conf association (see gap above)
gmc_method: none          # static camera; GMC('none') is a supported no-op (gmc.py:76-77)
proximity_thresh: 0.3     # from 0.5; opens the IoU gate so ReID can rescue the fast-moving teacher
appearance_thresh: 0.90   # strict: ReID overrides IoU only on near-certain matches; see T4 tuning note
with_reid: true
model: auto               # forward pre-hook on the Pose head, zero extra inference (trackers/track.py:60-74)
```

### Improvements

| ID | Change | Target | Rationale | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| T1 | Ship the yaml above; stop hardcoding `"botsort.yaml"` in two call sites. | new app/trackers/classroom_botsort.yaml, detector.py:`_track_frame`, config.py:`Settings.tracker_cfg` | All tracker tuning becomes deployable and env-overridable; every row below rides on this file. | Enables T2-T6. | low | None functionally. |
| T2 | `fuse_score: false` | yaml | Verified dead path when true (fixed 0.5 gate vs min achievable cost 0.75 in stage 2). Stage-1 loses conf weighting, which barely matters when seated IoU is near 1.0 frame to frame. | Back-row tracks survive conf dips instead of going lost and respawning; steadier occupancy (avg=25.5 should rise toward the true seated count); fewer raw tracks into merge. | low | Marginally more willingness to match low-quality boxes in stage 1; bounded by match_thresh. |
| T3 | `gmc_method: none` | yaml | Camera never moves; sparseOptFlow costs 10-30 ms/frame at this resolution (minutes over 20k frames) inside `_pre_first_associate`, serialized with MPS inference, and can warp Kalman states from person motion. | Minutes of CPU recovered per 68-min video; removes a drift source. | low | Zero on a verifiably static camera. |
| T4 | `with_reid: true, model: auto, proximity_thresh: 0.3, appearance_thresh: 0.90` | yaml | `model: auto` reuses the detector's own head features via forward pre-hook (verified: Pose subclasses Detect, head.py:558; yolo11 is not end2end, so the hook path is taken); no second model. In `get_dists` (bot_sort.py:189-202) ReID only overrides IoU inside the proximity gate, so at 0.90 uniform-colored students rarely trigger it while the teacher (distinct clothing, drifting Kalman prediction at 5 fps) gets rescued after occlusions. | Fewer teacher-track fragments (the dominant source of teacher_present_ms error) without new swap risk among look-alike students. | low | Head features are detection-oriented, weaker than a real ReID net; worst case at 0.90 is a no-op (falls back to IoU). Tuning note: research lenses proposed 0.65-0.90; start strict at 0.90, loosen toward 0.65 only if teacher fragments persist on the demo. If quality is insufficient, the one-dep upgrade is onnxruntime + yolo26n-reid.onnx via the existing ReID class (trackers/utils/reid.py, auto-downloads). |
| T5 | `track_buffer: 60` | yaml | Seated students have ~zero Kalman velocity, so a lost track's prediction stays planted at the desk; the reappearing detection re-IoUs the stale prediction and reactivates instead of spawning a new raw ID. 6 s (current) is shorter than typical occlusion episodes. | Direct cut in raw-track count (156 today for ~31 people) feeding merge; fewer identities to reconcile (42 today). | low | A departed student leaves a 12 s ghost; if another person takes the seat within 12 s the ID transfers. Re-verify entries=4/exits=4. |
| T6 | `new_track_thresh: 0.40` | yaml | Spawn gate only; low-conf back-row flicker still attaches to existing tracks but can no longer mint a raw ID from one noisy frame. | Fewer short-lived raw tracks; less fragment absorption in roles.py; fewer merge candidates. | low | A genuinely new person needs one frame >= 0.4 to get an ID; at the door (near field, large box) conf is high, so entry/exit is safe. Verify entries=4. |

### Quick wins

- `gmc_method: none` and `fuse_score: false`: two yaml lines, minutes of CPU saved and a verified-dead recovery path un-broken.
- `new_track_thresh: 0.40` + `track_buffer: 60`: two yaml lines that directly shrink the raw-track count.
- `with_reid: true` + `model: auto`: free teacher re-ID from the pose model's own features.
- Log `len(hists)` as `raw_track_count` in `detect_video` (folded into E6) so every yaml change is A/B-able against the baseline.

---

## 3. Re-ID / Identity Merge (merge.py)

### Gaps

- `_score_clusters` uses hist correlation ALONE (W_HIST=0.6) when both hists exist, no spatial gate. With 3 uniform colors across ~30 students, hist correlation between DIFFERENT students is ~0.9, so any two non-overlapping same-shirt fragments score ~0.55+ and chain into cross-room chimeras. `spatial_continuity` runs only when hists are absent, i.e. it is skipped exactly when it is needed.
- Appearance is never persisted: `db.replace_detections` writes only `{standing, back_to_camera, raw_track_id}` into meta, so `/rederive` (jobs.py:`remerge_from_raw`) silently degrades to spatial-only merging and can produce different identities than the original `/analyze`.
- No motion model: `spatial_continuity` compares raw endpoints, and `SPATIAL_BASE_TOL` was inflated 0.04 to 0.08 solely to reconnect the walking teacher, doubling the drift allowance for seated students, weakening the exact gate that must reject uniform chimeras.
- No seat-anchor concept: the strongest classroom prior (same seat = same student; both stationary at different seats = different students) is neither a bonus nor a veto.
- Greedy heap merge is order-dependent: one high-hist same-uniform pair can steal a fragment from its spatially correct successor.
- Candidate generation enumerates all n^2/2 pairs ignoring `MAX_GAP_MS`; fine at n=156 (0.12 s), a blow-up under live streams.

### Improvements

| ID | Change | Target | Rationale | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| M1 | Always compute `spatial_continuity(ca, cb, gap)` (centers already ride in every interval tuple). Re-score: `0.35*appearance + 0.25*spatial + 0.2*size + 0.2*temporal`, with the spatial term floored at 0.6 for mobile clusters (trajectory x/y range > 0.15, i.e. the teacher) so her long jumps still reconnect. | merge.py:`_score_clusters`, W_* weights | Under uniforms, appearance cannot distinguish students; endpoint geometry is the only discriminator for seated kids and today it is skipped when hists exist. | Eliminates same-shirt cross-room chimeras (wrong occupancy/max_students); makes MERGE_THRESHOLD=0.55 meaningful again. | low | Slightly fewer merges for genuinely same seated students with occlusion-corrupted endpoints; mitigated by M2. |
| M2 | Seat-anchor prior and veto: mark a cluster seated when bbox-center range < 0.02. For seated+seated pairs use median-center distance with tol 0.03 as the spatial term (endpoints are occlusion-corrupted, seat medians are not) and HARD-reject when median distance > 0.10 regardless of hist score. | merge.py:`_Cluster` (add median_center, center_range), `_score_clusters` | A seat is an identity anchor. Seat-swap is safe: swapping requires walking, those fragments exceed the 0.02 range and bypass both prior and veto. | Recovers fragmented seated students (avg_students accuracy) and gives an appearance-independent chimera veto; the single highest-leverage merge change for uniformed classrooms. | medium | A student who leans widely falls back to the normal path (no regression, just no bonus). |
| M3 | Add end/start velocity to `RawTrack` (mean displacement/s over last/first 3 samples, magnitude capped so v*gap <= 0.25). Score `exp(-(|last_center + v_end*gap - next_start| / tol)^2)` and revert `SPATIAL_BASE_TOL` 0.08 to 0.04. | merge.py:`RawTrack`, `build_raw_tracks`, `spatial_continuity` | The 0.08 base exists only because the walking teacher jumps 0.08-0.10 per ~1 s gap; linear extrapolation reconnects her at 0.04, restoring the tight gate seated students need. | Teacher stays one identity without loosening the student gate; fewer false merges after teacher-crossing occlusion cascades. | medium | Velocity is noisy at 5 fps and at track birth; the cap plus 3-sample averaging bounds the damage. |
| M4 | New table `track_appearance(video_id text, raw_track_id int, hist real[], embed real[], PRIMARY KEY(video_id, raw_track_id))`; write the median torso hist (960 floats) per raw track inside the `replace_detections` transaction (~156 rows/video, NOT per detection). `/rederive` fetches it and passes `{raw_id: [hist]}` into `merge.build_raw_tracks` instead of `{}`. | db.py (new fns), jobs.py:`run_pipeline`/`remerge_from_raw` | Histograms are computed during /analyze then thrown away; /rederive silently produces different identities. Per-track rows avoid bloating the 450k-row hypertable. | rederive reproduces analyze-quality merges; provides the column CLIP embeddings (M5) land in. | medium | Additive only; needs a drizzle migration. |
| M5 | CLIP ViT-B/32 track embeddings, zero new deps (openai-clip 1.0.1 already in the venv for YOLO-World). Alongside hist sampling (same <=10 samples/track cadence) save upper-60% bbox crops; at end of `detect_video` batch-embed on MPS (~1560 crops at batch 64, a few seconds per video); store L2-normalized median embedding per raw track in `track_appearance.embed`. In merge: `appearance = 0.5*cosine(embed) + 0.5*hist_corr` when embeds exist. | detector.py:`_extract_frame`, merge.py, track_appearance | HSV hists collapse under uniforms; CLIP separates by face/hair/build/glasses and is robust to CCTV auto-exposure shifts. Batched post-pass keeps the 5 fps loop untouched. | Appearance becomes discriminative BETWEEN same-uniform students; enables an embed-contradiction veto (cosine < ~0.35) for the seat-swap edge case. | medium | CLIP is not a person-ReID specialist; sub-40 px back-row crops embed poorly, so spatial gating stays co-evidence, never appearance alone. |
| M6 | Scale-proofing: stack all hists/embeds into a matrix, mean-center + L2-normalize once, `C = H @ H.T` (measured 0.02 s at n=1500 vs ~32 s of per-pair corrcoef); generate candidates by sorting on `first_ms` and bisecting the `[last_ms - 1000, last_ms + MAX_GAP_MS]` window instead of all pairs. | merge.py:`merge_tracks`, `hist_correlation` | No current bottleneck (0.12 s at n=156), but Kafka streams grow n without bound and per-pair corrcoef is the term that blows up. | Merge stays sub-second to n~5000; safe to lower tracker thresholds without merge-cost anxiety. | medium | None; numerically identical scores. |
| M7 | LAST, after M1-M3 land: reformulate as bipartite assignment (rows = fragment ends, cols = fragment starts, cost = -log(score), cost_limit = -log(MERGE_THRESHOLD)) with `lap.lapjv` (lap 0.5.13 already a dependency; detector.py ships a shim), chain matches transitively. | merge.py:`merge_tracks` | Greedy is order-dependent; assignment enforces each fragment start has at most one predecessor globally, exactly the constraint a reappearing person satisfies. Matters most when the teacher crossing a row fragments 3-4 students at once. | Removes greedy steal errors in occlusion cascades. | high | Behavior change; strict chains lose the heap's many-to-one absorption. Re-verify the full baseline before adopting. |

### Quick wins

- M1 weight fix is ~10 lines (`W_HIST` 0.6 to 0.35 plus an always-on 0.25 spatial term); immediately blunts uniform chimeras.
- Interim persistence hack before M4: stash the raw track's median hist into the meta JSON of that track's FIRST detection row in `replace_detections`, read it back in `fetch_detections`; closes the rederive gap in ~15 lines with zero schema change.
- Add a per-stage duration log line in jobs.py:`run_pipeline` (detect_s / merge_s / derive_s) so the "merge takes minutes" claim stays verifiably false (folded into E6).

---

## 4. Role Classification + Events (roles.py + events.py)

### Gaps

- Sitting teacher: with standing_ratio ~0.2 the composite lands ~0.53; if she is also never at the board it drops below `TEACHER_MIN_SCORE=0.5` and everyone becomes unknown. Fixed weights plus an absolute floor are not adaptive to per-video behavior distributions.
- `door_entry_exit` (events.py:91-102) checks only the single first/last sample of each presence interval; a tracker that locks on 1-2 s after the door places that sample mid-room and the real crossing is silently dropped.
- `events.derive` picks only the FIRST door zone (`next(... kind=='door')`); a second door is ignored entirely.
- `occupancy_buckets` counts distinct non-teacher track_nos per 5 s bucket, so an ID switch mid-bucket double-counts a student; max/avg are inflated by fragmentation rather than measured from concurrent boxes.
- `board_intervals_from_samples` resets run_start on ANY single false sample while off (lines 167-168), so opening requires ~10 consecutive true samples at 5 fps; one occlusion flicker discards a genuine board approach.
- Two teacher-like adults (a TA): the relative-margin rule fails and everyone becomes unknown, so teacher_present_ms=0 despite two clearly adult identities.

### Improvements

| ID | Change | Target | Rationale | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| R1 | Robust per-video z-score composite: per feature compute z = (x - median)/(1.4826*MAD + 1e-6) across eligible identities, clip to [0,3]; composite = weighted z sum with existing weights; teacher rule: best_z >= 2.0 AND (best - second) >= 1.0 z-units. Keep the fixed-weight path as fallback when eligible < 5. | roles.py:`composite_score`, `assign_roles` | With ~30 seated students the medians ARE student behavior, so a mostly-sitting teacher (0.2 vs median 0.05 standing) is still a multi-sigma outlier that the absolute 0.5 floor rejects today. Also fixes the never-at-board case (board z ~0 for everyone, other features carry). | Sitting-teacher and no-board videos classify correctly; demo result unchanged (true teacher is a larger outlier in z-space). | medium | Threshold recalibration; guard with the E4 regression run plus a small-clip test. |
| R2 | Window the door test: bisect the sorted teacher ts list and test `near_zone` on ALL samples in `[start, start+2000ms]` for entries and `[end-2000ms, end]` for exits (>=1 hit counts). | events.py:`door_entry_exit` | BoT-SORT often reacquires the teacher 1-2 s after the door. Windowing only rescues crossings at existing interval boundaries; it cannot invent events, so demo entries/exits can only stay equal or recover misses. | Entry/exit recall up on late-lock-on; zero new false positives by construction. | low | Keep the window <= 2 s so a mid-room reappearance near the door cannot count. |
| R3 | Multiple doors: collect `door_polygons = [z['polygon'] for z in zones if kind=='door']`; near-door = `any(near_zone(det, p) for p in door_polygons)`. | events.py:`derive`, `door_entry_exit` | Rooms with front+back doors currently lose every crossing through the second door. | Correct counts in two-door rooms; no-op for single-door videos. | low | None. |
| R4 | Fragmentation-immune occupancy: per bucket, group non-teacher detections by `video_ts_ms` and set students = max (or 90th percentile) over frames of the per-frame box count, instead of `len(set(track_no))`. | events.py:`occupancy_buckets` | Concurrent boxes can never double-count one person; identity-free counting decouples the dashboard headcount from re-ID quality. | Occupancy stops inflating as merge quality varies. Validate against max=30 / avg=25.5. | medium | Per-frame misses (occluded back rows) now lower counts within a bucket; hence max/percentile per bucket, not mean. |
| R5 | 1 s majority filter: smooth each track's chronological standing booleans with a 5-sample sliding majority vote before computing standing_ratio and the at-board standing gate. | roles.py:`compute_features` | `_is_standing` flips on bbox-aspect flicker around 1.6 and keypoint dropouts when the teacher writes back-turned; posture changes last >> 1 s so a majority vote beats an HMM at zero complexity. Complements D4 upstream. | Cleaner standing separation between students and teacher; board_proximity stops losing samples to single-frame dropouts. | low | None material. |
| R6 | Flicker-tolerant board open: keep a false_streak counter; only reset run_start when false_streak >= 2 samples (or false time >= 600 ms); open when `ts - run_start >= on_ms` with total false time <= 600 ms. | events.py:`board_intervals_from_samples` | Opening currently needs ~10 CONSECUTIVE true samples; one occluded frame discards a genuine approach, so short board stints never register. The off side (3 s) already tolerates flicker; the on side does not. | Recovers short genuine board intervals. Re-verify against board_ms=41200 and tighten the false-time budget if it overshoots. | low | Slight board_ms increase expected; bounded by the budget. |
| R7 | Assistant role instead of all-unknown collapse: when the margin test fails but BOTH top candidates clear min_score AND second leads third by `max(abs_margin, rel_margin*second)`, assign best=teacher, second=assistant. Keep presence/board on the teacher only; exclude assistant from student counts in `occupancy_buckets`. | roles.py:`assign_roles`, events.py:`derive`, `occupancy_buckets` | A TA is the exact case the relative-outlier rule punishes hardest: two similar high scores, margin fails, teacher_present_ms=0 and the TA counts as a student. The third-place check preserves graceful degradation for genuinely ambiguous crowds. | Two-adult classrooms produce teacher analytics instead of zeros. | medium | May label a hyperactive presenter as assistant; only affects occupancy exclusion, not teacher metrics. |
| R8 | Sustained-locomotion movement: split each track into 60 s windows; movement = fraction of windows with center range > 0.1 (patrol ratio), blended 50/50 with the current global-extent feature. | roles.py:`compute_features`, `MOVEMENT_RANGE_NORM` | A student who walks up once to present matches the teacher's global x_range; the teacher's signature is REPEATED locomotion. Presenter: 1-2 active windows; teacher: most windows. | Presenters stop threatening the margin; teacher lead widens. | medium | A lectern-bound teacher scores lower on patrol; the 50% blend keeps presence/standing dominant. |

### Quick wins

- R2, R3, R6 are each a handful of lines and pure recall fixes.
- R5's majority filter is ~5 lines in `compute_features`.
- Persist per-identity feature vectors + composite scores into analytics (a `debug` key in jobs.py:`derive_result`) so sitting-teacher/TA thresholds can be tuned from real runs instead of reruns.

---

## 5. Evaluation Harness (LAND FIRST)

Nothing in sections 1-4 is safely shippable without this: all 109 existing tests assert shapes/thresholds, zero measure accuracy, and the verified baseline exists only as remembered numbers.

### Gaps

- No golden JSON, no regression run; any merge/roles tweak can silently regress the verified numbers.
- Only pipeline fixture is tests/test_integration.py:`_synthetic()` (2 people, no occlusion, no uniforms, no second adult); the render script makes one static frame with no people.
- No merge-quality signal without GT (fragments-per-identity, chimera formation computed nowhere).
- Only perf test is 250 fragments / 10 s; real footage has far more raw tracks and the all-pairs `_push_pairs` loop (merge.py:327-329) is untested at scale.
- No GT format or annotation path for real CCTV clips, so the uniform re-ID weakness cannot be quantified.

### Improvements

| ID | Change | Target | Rationale | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| E1 | `evals/` package + pure metric library: `interval_sym_diff_ms` (|union|-|intersection|, catches offsetting errors that |sum-sum| hides), `event_prf` (greedy per-kind matching, tol 5000 ms), `occupancy_mae` (GT timestamps vs containing 5 s bucket), `teacher_anchor_accuracy` (IoU >= 0.3 at anchor timestamps), `anchor_fragmentation` + `id_switches` from sparse anchors, and no-GT proxies (n_identities, mean fragments/identity, orphan_mass = detection fraction in identities spanning < 60 s). | new services/ml-service/evals/metrics.py | Every dashboard number is currently unmeasured; every other harness piece consumes these functions. Sparse anchors give MOT-like signal without per-frame labels. | Each accuracy change becomes a number. | low | None; pure functions over existing dicts. |
| E2 | Pydantic-validated GT schema v1 for short annotated clips: clip meta, zones, teacher present/board intervals, door events (tolerance defaulting to PRESENCE_GAP_MS=5000 since exits stamp at last detection), teacher anchors ~1/10 s, occupancy samples every 30 s, optional identity anchors for 5-8 students. bbox normalized top-left, matching `Detection.bbox`. | evals/gt_schema.py, evals/gt/*.gt.json | Sparse anchors make a 2-min 30-person clip annotatable in ~10 min yet suffice for teacher-ID accuracy and fragmentation/IDSW. | 3-5 annotated real-CCTV clips turn the uniform re-ID weakness into tracked numbers. | low | Anchor alignment: match nearest sampled detection within +/-200 ms. |
| E3 | Detection-level scenario generator with exact built-in GT (generalizes `_synthetic()`): uniforms (30 students sharing 3 hist archetypes), occlusion_refragment (3-8 s sample drops, new raw_id, ~0.02 endpoint offset), teacher_sitting, two_adults, seat_swap (same-hist students swap seats during overlapping 6 s gaps; merge must NOT chain them). Assert purity==1.0 on seat_swap, fragments/person <= 2 on occlusion, correct teacher on two_adults. | evals/scenario_gen.py, tests/test_scenarios.py | merge/roles/events are pure functions of Detection streams; synthesizing at Detection level yields exact identity GT (impossible from rendered video) and runs in ms without GPU. | Exact purity/IDSW measurement for exactly this classroom's failure modes; regression guard for MERGE_THRESHOLD, SPATIAL_BASE_TOL, W_*. | medium | Synthetic can be too clean; calibrate jitter/conf from the E4 fixture. |
| E4 | Real-video regression harness: `snapshot.py` exports `db.fetch_detections` + `fetch_video_info` to `fixtures/demo.dets.jsonl.gz`; `regress.py` replays `jobs.remerge_from_raw` + `jobs.derive_result` fully offline and diffs analytics vs `demo.golden.json` carrying its own tolerances (present/board +/-5000 ms, entries/exits exact, max +/-1, avg +/-1.0, n_identities +/-5; timing budgets remerge < 30 s, derive < 10 s); `--bless` rewrites golden; pytest marker `regression`, auto-skip when fixture absent. | evals/snapshot.py, evals/regress.py, evals/fixtures/, tests/test_regression_real.py | remerge_from_raw is exactly the /rederive production path; replays real footage in seconds instead of a 30-40 min YOLO re-run. | Every merge/roles/events change gets pass/fail against verified real-footage numbers in < 1 min. | medium | 31k-det fixture is ~3-6 MB gz (committable); full 450k export stays local-only with CI skip. |
| E5 | Merge timing-budget test at realistic fragment count: load raw tracks from the real fixture (fallback: scenario_gen at ~2500 raw tracks), assert `merge_tracks` < 30 s and `derive_result` < 10 s, print n_raw_tracks and pairs scored. Marker `perf`. | tests/test_merge_perf.py | Makes M6 verifiable instead of vibes; catches future O(n^2) regressions before Kafka-live latency. | Locks merge cost. | low | Wall-clock flakiness on CI; 2x bound via env var. |
| E6 | Per-run diagnostics on EVERY pipeline run: optional `diagnostics` field on `AnalysisResult` with `{n_detections, n_raw_tracks, n_identities, orphan_mass, detect_s, merge_s, derive_s}` populated in `run_pipeline`; raw_track_count is free (`len(hists)` in `detect_video`). | app/jobs.py:`run_pipeline`, app/models.py:`AnalysisResult`, detector.py:`detect_video` | Fragmentation ratio and stage latency become visible on every production run including live streams, with zero GT. Absorbs the tracking lens's raw_track_count item and the merge lens's stage-timing item. | A regression (identities 42 to 120, merge_s > 30) is caught on the first real upload. | low | Field must be Optional default None; confirm dashboard ignores the extra key. |
| E7 | Eval runner CLI + append-only ledger: `uv run python -m evals.run_eval scenarios | regress --fixture demo | clip --gt ...`; each run appends `{git_rev, timestamp, mode, metrics...}` to evals/results/history.jsonl. Makefile targets. | evals/run_eval.py | Accuracy work needs a before/after ledger across commits; jsonl diffs cleanly in PRs. | Metric trend per commit. | low | None. |
| E8 | cv2 anchor-annotation helper (local, non-CI): step a clip at 10 s intervals, click-drag teacher anchors, key-toggle present/board boundaries and door stamps, optional student-anchor mode; writes schema-valid JSON. | evals/annotate.py | A schema without tooling never gets populated; 10 min/clip is the difference between 5 GT clips and none. | Unblocks real-footage measurement within a day. | medium | GUI tool, keep out of pytest collection. |

### Quick wins

- Snapshot the demo video's detections from TimescaleDB (localhost:5433) TODAY (~40 lines over existing db fns); a reanalyze would overwrite the `meta.raw_track_id` state the harness needs.
- Check in `demo.golden.json` with 161800/41200/4/4/30/25.5/42 + tolerances now; the baseline becomes data before any harness code lands.
- Land `metrics.py` first (~40 lines for the two core functions).
- Add pytest markers `regression`/`perf` with `addopts = -m 'not regression and not perf'` so the fast suite stays fast.
- Lift `_synthetic`/`_teacher_det`/`_student_det` into tests/helpers.py so scenario_gen starts from proven builders.

---

## 6. Storage + Live-Stream Tiering (TimescaleDB)

### Gaps

- The hypertable is partitioned on per-video-relative `video_ts_ms` (integer, chunk_time_interval=3600000): every video shares the same 0..68 min chunk space, chunks never age, and `add_retention_policy`/`drop_chunks`/`add_compression_policy` cannot work because no wall-clock dimension exists (drizzle/0001_hypertable.sql).
- `bbox` and `meta` are jsonb per row (~130 of ~185 bytes/row are repeated JSON keys); jsonb defeats columnar delta/gorilla encoding, capping even manual compression at ~3-5x instead of 10-20x.
- Zero compression/retention/cagg policies: ~1.7 GB/day/camera at 5 fps live; the only aggregate is a batch-recomputed jsonb blob.
- `replace_detections` is delete-then-COPY of the entire video in one transaction: incompatible with append-only live ingest and with compressed chunks (DELETE forces decompression).
- No backpressure or sub-job idempotency: unbounded `queue.Queue`, idempotency at whole-video granularity only.

### The tiering model

Three tiers (the Milestone/Genetec/DeepStream persistence split):

1. **Hot ring buffer**: raw per-frame `detection_events`, compressed after 24 h, dropped after 7 days. Enough to rederive, audit, and serve full-fps overlay for recent video.
2. **Permanent overlay tier**: per merged track, RDP-simplified `(ts, cx, cy)` polylines (epsilon 0.005 normalized) plus one full bbox keyframe every 2 s, stored in the existing `tracks.meta` jsonb during `derive_result`. ~42 tracks x 50-200 points + ~2000 keyframes/track-hour replaces 450k rows at ~2% of the size, < 2 MB/video. `getDetections` serves keyframes when raw rows have aged out.
3. **Permanent aggregates**: events, track summaries, `video_analytics`, and a 1-minute occupancy continuous aggregate. Kept forever at negligible size.

Net effect: detection_events caps at ~0.5-1 GB compressed per camera instead of unbounded 1.7 GB/day, and everything the dashboard shows survives retention.

### Exact SQL

Step 0, safe today before any re-key (plain ALTER, instant with a default in PG11+; gives the later migration real data to partition on, and lets VOD chunks compress now since they are static post-write):

```sql
ALTER TABLE detection_events ADD COLUMN ts timestamptz NOT NULL DEFAULT now();

ALTER TABLE detection_events SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'video_id',
  timescaledb.compress_orderby   = 'video_ts_ms, track_no'
);
-- after each job reaches done:
SELECT compress_chunk(c, true) FROM show_chunks('detection_events') c;
-- decompress_chunk before any reanalyze/rederive DELETE.
```

Step 1, re-key the hypertable on wall clock (maintenance window; `migrate_data` takes a table lock). Add `stream_id` in the same migration for Kafka. Keep `video_ts_ms` as an ordinary column and the `(video_id, video_ts_ms)` index for VOD reads:

```sql
SELECT create_hypertable(
  'detection_events', 'ts',
  chunk_time_interval => INTERVAL '1 hour',
  migrate_data        => TRUE
);
```

Step 2, typed columns in the same PR (one migration, one read-path change in `getDetections` and `/rederive`): `bbox jsonb` becomes `x, y, w, h smallint` quantized to 1/10000 of frame (< 0.3 px error at 2560 px); `meta jsonb` becomes `standing bool, back_to_camera bool, raw_track_id int`. ~185 B/row to ~60 B uncompressed; typed ints/bools get delta-of-delta/simple8b under columnar compression (10-20x total, vs ~3-5x for jsonb).

Step 3, hot-tier policies:

```sql
ALTER TABLE detection_events SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'video_id',
  timescaledb.compress_orderby   = 'ts, track_no'
);
SELECT add_compression_policy('detection_events', INTERVAL '24 hours');
SELECT add_retention_policy('detection_events', INTERVAL '7 days');
```

Caveats: `/rederive` and reanalyze DELETEs hit compressed chunks; either `decompress_chunk` that video's chunks first or restrict rederive to the < 24 h window. Retention ends rederive for old videos, acceptable because tiers 2-3 preserve everything the dashboard shows.

Step 4, occupancy continuous aggregate (caggs forbid `count(DISTINCT)`, so two levels; refresh lag 1 h stays 6.9 days ahead of retention, so aggregates always materialize before raw drops):

```sql
CREATE MATERIALIZED VIEW track_minute
WITH (timescaledb.continuous) AS
SELECT video_id,
       time_bucket(INTERVAL '1 minute', ts) AS bucket,
       track_no,
       count(*) AS n
FROM detection_events
GROUP BY 1, 2, 3
WITH NO DATA;

CREATE VIEW occupancy_minute AS
SELECT video_id, bucket, count(*) AS bodies
FROM track_minute
GROUP BY 1, 2;

SELECT add_continuous_aggregate_policy('track_minute',
  start_offset      => INTERVAL '1 hour',
  end_offset        => INTERVAL '2 minutes',
  schedule_interval => INTERVAL '1 minute');
```

(Alternative: `timescaledb_toolkit` `approx_count_distinct` in one level. Teacher/student split joins role from `tracks` at query time.)

### Quick wins

- `replace_detections` meta dict: omit false-valued keys (write standing/back_to_camera only when True, raw_track_id only when != track_no); `fetch_detections` already defaults via `.get()`. 30-40 bytes saved on the vast majority of 450k rows, 3 lines.
- `jobs.py: _queue = queue.Queue(maxsize=4)`: one-line backpressure so a burst of /analyze submissions blocks (or 429s) instead of holding N videos in memory.
- Write the RDP polyline into `tracks.meta` during `derive_result` now: zero schema migration, and the permanent overlay tier exists before any retention policy turns on.

---

## 7. Kafka Readiness

Ordering matters: harness (5) first, then storage re-key (6), then these, because live correctness is unverifiable without the diagnostics and the append path needs the `ts`/`stream_id` columns.

| ID | Change | Target | Rationale | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| K1 | Frame-source abstraction: extract `iter_frames(source) -> Iterator[tuple[ts_ms, ndarray]]` with a FileSource (current grab/retrieve/stride loop, keeps `_validate_video_path`) and later a KafkaSource; `detect_video` becomes `process_frames(frames, state)`. Move `_reset_tracker` to stream start only; replace module-global `_model`/`_fallback_cpu` with a per-stream context holding its own `ultralytics.trackers.bot_sort.BOTSORT` instance so N cameras share one process. | detector.py:`detect_video` + new FrameSource protocol | `_extract_frame`/`_track_frame` are already frame-granular; only the enclosing while-loop and singletons couple the pipeline to a local file. This is the single seam Kafka needs. | Same detection code for uploads and Kafka/RTSP; multi-camera per process; FileSource reproduces today's stride/ts math exactly, so zero VOD behavior change. | medium | track_buffer semantics are sampled-frames (12 s at 5 fps with the section 2 yaml); keep that in mind for gap tolerance. |
| K2 | Windowed online merge + streaming event state: process in T=30 s windows; maintain an active-identity gallery (last detection within 60 s); score new/ended fragments only against the gallery. Events carries `{last_teacher_ts, board_hysteresis_state, open_presence_interval, partial_occupancy_bucket}` between windows, emitting finalized events and 1-min aggregates per flush. | merge.py (new `merge_window`), events.py (new StreamState), jobs.py:`run_pipeline` | Current merge needs the whole video and events needs the full timeline; gallery-windowed matching is the standard online pattern (DeepStream/Metropolis: per-window metadata to broker to event store). | Live latency bounded to ~T + hysteresis (~35-40 s); merge cost per window O(active^2) ~ 40^2 instead of global; constant memory. | high | Windowed merge can split identities a global pass would join; mitigate with a nightly batch /rederive over the 7-day raw tier as reconciliation. |
| K3 | Append-only ingest with per-window ledger + bounded backpressure: `append_detections(stream_id, window_seq, rows)` COPYs the window AND inserts into `ingest_ledger(stream_id, window_seq PRIMARY KEY, kafka_offset, row_count)` in one transaction; replay skips on ledger hit; commit Kafka offsets only after the DB txn (offsets-in-DB pattern). `queue.Queue(maxsize=3)` windows with partition pause()/resume() on watermarks. Keep `replace_detections` + run fences (VideoDeletedError/StaleRunError) for the VOD path only. | db.py (new `append_detections`), jobs.py:`_queue`, future Kafka consumer | Delete-swap and video-level fences are VOD constructs; live needs idempotent append and an anchor for at-least-once delivery so a crash between COPY and offset commit cannot duplicate rows. | Zero duplicate detections across rebalances/restarts; GPU stalls back-pressure Kafka instead of OOMing the worker. | medium | One small ledger insert per 30 s window (negligible); needs `stream_id` from the section 6 re-key migration. |

---

## 8. Prioritized Roadmap

Ordered by impact/effort. Measurement items first: accuracy work is unmeasurable without them. Each item lists the gate it must pass.

| P | Item | Node | Effort | Impact | Gate |
|---|---|---|---|---|---|
| 1 | Baseline as data: snapshot demo detections from TimescaleDB + check in demo.golden.json + evals/metrics.py (E1 core) | EVAL | low | Blocks everything; a reanalyze destroys the fixture state | Fixture + golden committed |
| 2 | Per-run diagnostics: raw_track_count, n_identities, orphan_mass, detect_s/merge_s/derive_s on AnalysisResult (E6, absorbs T-metric + M-timing items) | EVAL+TRACKING+RE-ID | low | Every later change gets a free A/B signal on every run | Dashboard tolerates the extra key |
| 3 | Offline regression harness: regress.py replaying remerge_from_raw + derive_result vs golden with tolerances (E4) + pytest markers | EVAL | medium | Real-footage pass/fail in < 1 min instead of 40-min re-runs | Green on unmodified code |
| 4 | Detection kwargs: imgsz=1280, half=True (mps only), conf=0.05, max_det=100 (D1-D3) | DETECTION | low | Largest single quality lever; back-row recall and keypoint validity | Baseline within tolerance; wall time <= ~1.5x |
| 5 | classroom_botsort.yaml: gmc none, fuse_score false, buffer 60, new_track 0.40, track_low 0.05, with_reid auto (T1-T6) | TRACKING+DETECTION+RE-ID | low | Un-breaks stage-2 recovery, kills GMC waste, cuts raw tracks feeding merge, free teacher ReID | raw_track_count drops from 156; entries/exits stay 4/4 |
| 6 | Merge scoring fix: always-on spatial term, weights 0.35/0.25/0.2/0.2, mobile-cluster floor (M1) | RE-ID | low | Kills same-shirt cross-room chimeras that corrupt occupancy | n_identities within 42 +/- 5; scenario purity |
| 7 | Events recall bundle: door window +/-2 s, multi-door list, board flicker tolerance (R2, R3, R6) | ROLES+EVENTS | low | Pure recall fixes, cannot add false events by construction | entries/exits >= baseline, never above GT; board_ms near 41200 |
| 8 | Standing signal cleanup: STANDING_KPT_CONF=0.4 + 90 px box gate (D4) + 1 s majority filter (R5) | DETECTION+ROLES | low | Stabilizes the 0.30-weighted teacher feature end to end | Teacher margin widens on demo |
| 9 | Scenario generator with exact GT: uniforms, seat_swap, occlusion, two_adults, teacher_sitting (E3) | EVAL | medium | Exact purity/IDSW for exactly this room's failure modes; guards items 6, 10, 11 | seat_swap purity 1.0 on current+new code |
| 10 | Seat-anchor prior and veto for stationary clusters (M2) + velocity extrapolation with SPATIAL_BASE_TOL back to 0.04 (M3) | RE-ID | medium | Strongest classroom prior; appearance-independent chimera veto; teacher reconnects without loosening the student gate | Regression + seat_swap/occlusion scenarios green |
| 11 | Robust z-score teacher selection + assistant role (R1, R7) | ROLES | medium | Fixes sitting-teacher and TA collapse (teacher_present_ms=0 cases) | Demo teacher unchanged; two_adults scenario green |
| 12 | Fragmentation-immune occupancy via concurrent per-frame box counts (R4) | EVENTS | medium | Decouples headcount from re-ID quality entirely | max=30, avg ~25.5 on demo |
| 13 | Appearance persistence (track_appearance table, M4) + CLIP ViT-B/32 track embeddings (M5) | RE-ID | medium | rederive stops silently degrading; appearance becomes discriminative between same-uniform students | rederive == analyze identities on demo |
| 14 | Storage tiering: ts column + re-key hypertable + typed columns + compression/retention + occupancy cagg + RDP overlay tier (section 6) | STORAGE | medium | Unbounded 1.7 GB/day/camera becomes a capped ring buffer; dashboard survives retention | Policies active; getDetections serves aged videos from keyframes |
| 15 | Kafka seam: FrameSource + per-stream tracker context (K1), then windowed merge/events (K2) and ledgered append ingest (K3) | STORAGE/LIVE | high | Live streams with bounded ~35-40 s latency, multi-camera, exactly-once-ish ingest | FileSource bit-identical VOD results; replay produces zero duplicates |

Deferred/decide-later: greedy-to-lapjv global assignment (M7, only after 6 and 10 land), vectorized correlation + windowed candidates (M6, when raw-track counts approach ~1000 under live), yolo11l-pose benchmark (D6), tiling (rejected, D5), annotation tool + GT clips (E8/E2, schedule alongside item 9 to get real-footage numbers).
