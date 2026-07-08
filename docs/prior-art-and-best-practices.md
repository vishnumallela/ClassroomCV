# Luminary — Prior Art & Best Practices

**Status:** committable engineering + product guidance
**Scope:** maps every recommendation from four prior-art research lenses onto our exact stack
**Our stack (as of this doc):** static wide-angle CCTV → YOLO26-pose (17 kpts, NVIDIA GPU) → BoT-SORT (`gmc none`, `with_reid false`, `track_buffer` per `trackers/classroom_botsort.yaml`) → identity merge via CLIP ViT-B/32 median embeddings + HSV torso histograms + hard seat-anchor spatial veto + **greedy heap merge** (`merge.py`) → rule-based teacher classification from four hand-weighted signals (standing 0.30 / movement 0.25 / presence 0.25 / board 0.20, `roles.py`) with an outlier-margin rule → SAM2 + YOLO-World zone detection → TimescaleDB event store → additive data-quality confidence report (`quality.py`).

---

## 1. Executive summary — the highest-leverage changes

Prior art points to five moves, in order of leverage:

1. **Replace CLIP-general + HSV appearance with a person-ReID backbone (CLIP-ReID ViT-B or SOLIDER), fine-tuned on each room's own pseudo-labelled gallery.** On MSMT17 — the hardest same-appearance benchmark — CLIP-ReID hits 73.4–75.8 mAP and SOLIDER ~77.1 vs OSNet 52.9; general CLIP and HSV color histograms are not even on that curve, because a uniformed classroom collapses "a student in a blue shirt" to one vector and HSV to one color (CLIP-ReID, arXiv:2211.13977; SOLIDER, arXiv:2303.17602). This is the single biggest lever we have.
2. **Add motion-first, appearance-free association cues to BoT-SORT — OC-SORT's Observation-Centric Re-Update/Momentum and Hybrid-SORT's weak cues (detection-confidence state, height/aspect state, velocity direction).** On DanceTrack (the governing same-uniform benchmark) these carry HOTA from BoT-SORT 53.8 to OC-SORT 55.1 to Hybrid-SORT 61–66 with _no_ appearance model — exactly our regime where appearance is degenerate (OC-SORT, CVPR2023; Hybrid-SORT, arXiv:2308.00783).
3. **Replace our greedy heap merge with an offline, per-lesson global tracklet association (GTA-style): split impure tracklets by appearance-cluster inconsistency, then re-link fragments with a two-stage Hungarian under a hard spatiotemporal-conflict constraint.** GTA lifted SORT +10.24 HOTA and Deep-EIoU+GTA reached 81.04 HOTA on SportsMOT, cutting ID switches (GTA, arXiv:2411.08216). Greedy heap merging cannot enforce that two tracks on screen _at the same instant_ are different people — our #1 structural failure mode.
4. **Add a one-time ground-plane homography so proximity, dwell and speed are reported in metres, not pixels — projecting the mid-ankle keypoints YOLO26-pose already gives us.** Pixel-space metrics are camera-dependent and non-comparable across rooms; Moodoo's whole validated feature set (1 m proximity, 10 s dwell) only means anything in world coordinates (Moodoo, PMC7334189; UCMCTrack, arXiv:2312.08952).
5. **Rename every metric to a behavioural/geometric observable and hard-delete "engagement / attention / affect / emotion" from schema, UI, DB and API.** The construct is scientifically invalid from video (Barrett et al. 2019) and inferring students' attention/motivation via biometrics is _prohibited_ in EU education under AI Act Art. 5(1)(f). This is a low-effort, non-negotiable legal safe-harbor move.

Items 1–3 harden the per-student time series our entire product rests on; item 4 makes cross-room claims honest; item 5 keeps us shippable.

---

## 2. What the field does that we already do well

Our existing choices are well-supported by the literature — worth stating so we don't regress them:

- **Edge/on-prem GPU processing, events-only persistence.** EduSense — the reference deployed classroom pipeline — processed 4K streams on-prem, stored _no_ video, and identified no individuals (Ahuja et al., IMWUT 2019; CMU SCS 2019). Our self-hosted GPU + TimescaleDB event store (no frame archive) matches this defensible pattern directly. Keep it; codify frame-discard with a TTL.
- **Rule-based, temporally-persistent teacher classification instead of pose-alone.** The reference pose paper (AlphaPose + Faster R-CNN, PMC12193412) never specified how teacher is distinguished from student; the SCB detector line (arXiv:2310.02522) doesn't track at all. Our geometry+mobility+persistence rule (`roles.py`, `teacher_chain.py`) is _ahead_ of the published detection work on this specific problem — the id-steal stitching in commit `d8489b6` is exactly the right instinct.
- **BoT-SORT with a second low-score association pass, GMC off for a bolted-down camera.** ByteTrack's low-confidence second-pass association is the widely-cited fix for identity preservation through occlusion (arXiv:2110.06864); turning GMC off for a static camera is correct — the yaml's reasoning (walking teacher biases optical-flow GMC) is sound.
- **A hard seat-anchor spatial veto in the merge.** Encoding "two stationary tracks at different desks cannot be the same identity" is a primitive form of the spatiotemporal-conflict constraint GTA formalizes (arXiv:2411.08216). Right idea; §3 generalizes it.
- **Keypoint-based (not face-based) identity, YOLO26-pose 17-keypoint skeletons.** ClassID shows pose/appearance re-ID sustains longitudinal continuity _without_ faces (Patidar et al., IMWUT 2024); privacy-preserving pose pipelines carry no PII (arXiv:2403.17175). Our skeleton-first design is the ethically correct substrate.
- **An additive data-quality confidence report (`quality.py`).** Reporting detection/tracking confidence rather than a bare percentage is exactly what the engagement literature says is the _only_ validated thing to report (detection accuracy, not the construct). Keep and extend it (see §5, §6).

---

## 3. Tracking & re-ID under uniforms — ranked upgrade path

Same-uniform tracking is a _solved-but-caveated_ regime (DanceTrack, SportsMOT). Our appearance stack is the weakest link because CLIP-general + HSV are the two things that collapse when everyone wears the same thing. Ranked by impact/effort:

### 3.1 (Highest) Swap CLIP-general + HSV → a person-ReID backbone, fine-tuned per room

**Why:** On MSMT17 mAP — CLIP-ReID ViT 73.4→75.8 (with SIE+OLP), SOLIDER Swin-B ~77.1, TransReID 64.9, OSNet 52.9; general CLIP-image / HSV are far below (CLIP-ReID, arXiv:2211.13977; SOLIDER, arXiv:2303.17602; TransReID, arXiv:2102.04378). Person-ReID features encode fine body/build/gait cues that survive identical uniforms; CLIP embeds color+coarse semantics and HSV is _pure_ color — useless when color is identical.
**Even bigger lever — in-domain fine-tuning:** CLIP-ReID trains **without concrete text labels** (two-stage: learn per-identity prompt tokens with frozen encoders, then fine-tune the image encoder). Mine high-confidence single-identity tracklets from BoT-SORT as pseudo-IDs, learn prompts, fine-tune, and rebuild the gallery **per room/camera** (arXiv:2211.13977).
**Drop-in point:** `merge.py::build_raw_tracks` currently computes `hist` (median torso HSV) and `embed` (median CLIP). Replace the embedding source with a CLIP-ReID/SOLIDER embedding exported to **TensorRT FP16** for the GPU (mirrors our YOLO26 `.engine` export path in `config.py`). Keep the median-over-samples aggregation and L2-normalization already there. Demote HSV to a tertiary tie-breaker or drop it; the merge's `EMBED_VETO_COS` and appearance-credit mapping stay but now operate on discriminative features.
**Effort:** medium · **Impact:** high · Keeps the CLIP-family backbone → minimal code churn.

### 3.2 Add OC-SORT + Hybrid-SORT motion/weak cues to BoT-SORT association

**Why:** These are cheap, appearance-free, and carry association precisely when uniforms match. OC-SORT's **Observation-Centric Re-Update (ORU)** corrects accumulated Kalman error on re-appearance; **Observation-Centric Momentum (OCM)** adds velocity-direction consistency; it beats ByteTrack by >10 HOTA on DanceTrack with no appearance model (OC-SORT, CVPR2023). Hybrid-SORT adds **detection-confidence state + height/aspect state + velocity direction** for +7.6 HOTA over OC-SORT (arXiv:2308.00783). Build/height and motion direction are the discriminators that remain when color/texture are identical.
**Drop-in point:** These live in the online tracker, not the merge. Our tracker is Ultralytics BoT-SORT via `trackers/classroom_botsort.yaml`. Options: (a) fork the BoT-SORT Kalman/association to add ORU/OCM + weak-cue costs, or (b) swap to an OC-SORT/Hybrid-SORT implementation behind the same `detector.py` track interface. The yaml note that `with_reid true` ID-switched the teacher is consistent with the literature — the fix is _not_ appearance ReID in the online stage, it's these motion cues.
**Effort:** medium · **Impact:** high.

### 3.3 Replace greedy heap merge → GTA-style offline global tracklet association

**Why:** `merge.py` does greedy agglomerative merging (`heapq` at the merge loop, threshold 0.55): pick the globally-highest-scoring pair, commit, repeat. This has **no whole-trajectory reasoning and no temporal-conflict guard** beyond the seated-anchor veto — so it can merge two students who are simultaneously on screen if their appearance agrees. GTA reasons over entire tracklets: (1) **Tracklet Splitter** clusters each tracklet's per-frame ReID embeddings and splits where cluster identity is inconsistent (this catches the tracker id-steals commit `d8489b6` hand-stitched); (2) **Connector** re-links fragments with a two-stage Hungarian over _averaged_ per-tracklet embeddings, subject to a **spatiotemporal-conflict constraint** (time-overlapping tracks on a static camera are different identities). GTA lifted SORT +10.24 HOTA; Deep-EIoU+GTA hit 81.04 HOTA on SportsMOT (the closest analog to a same-uniform classroom), cutting ID switches 2909→2737 (arXiv:2411.08216).
**Drop-in point:** `merge.py::merge_tracks` is already an offline pass over `RawTrack`s with per-track median embeddings — it's structurally _most of the way_ to GTA. Concretely: (i) generalize `SEATED_VETO_DIST`/anchor veto into a first-class mutual-exclusion constraint on **any** time-overlapping cluster pair (`_overlap_ms` already exists); (ii) add a splitter stage that runs _before_ merging, clustering per-frame embeddings within a raw track; (iii) replace the greedy heap with a windowed two-stage Hungarian; (iv) store **per-tracklet averaged** ReID embeddings (more stable than any single frame under uniforms) as the merge key. Since we already batch to TimescaleDB, a per-lesson offline pass fits naturally.
**Effort:** high · **Impact:** high.

### 3.4 Ground-plane Mahalanobis association (UCMCTrack) — pairs with §5 homography

**Why:** UCMCTrack runs the Kalman filter on the **floor plane** and replaces IoU with a **Mapped Mahalanobis Distance**; motion-only, it reaches 63.6 HOTA on DanceTrack (arXiv:2312.08952). Two seated students at different depths whose _image boxes overlap_ are far apart in floor coordinates — IoU confuses them, ground-plane distance doesn't. Camera-motion compensation is trivially identity for our bolted-down camera.
**Drop-in point:** Requires §5's homography first. Then feed the per-detection ground-plane covariance (§5.3) into the association cost. Composes with §3.2.
**Effort:** medium · **Impact:** medium (upgrades once homography lands).

### 3.5 Low-effort drop-ins: Deep-EIoU + BoostTrack++ tricks

- **Deep-EIoU (Expansion-IoU):** iteratively expand unmatched boxes before Hungarian — recovers matches for fast/jumping students where IoU falls to zero; SportsMOT-SOTA association step (arXiv:2602.00484). Low effort, drop-in to the association round.
- **BoostTrack++ trio (MOT-agnostic, "usable in any MOT algorithm"):** soft-BIoU (scale spatial cost by tracklet confidence), a shape-similarity term, and a **soft detection-confidence boost** that revives likely-occluded low-score detections + a relaxed similarity threshold for stale tracklets. #1 online HOTA on MOT17/MOT20 (66.6/66.4) (arXiv:2408.13003). Targets our exact failure: a briefly-occluded student dropped by a hard threshold re-enters as a new ID.
  **Effort:** low · **Impact:** medium.

**Ordering rationale:** §3.1 (ReID backbone) and §3.2 (motion cues) are independent and both attack the degeneracy directly — do them first. §3.3 (GTA) subsumes and generalizes our seat veto — do it once the embeddings feeding it are discriminative (i.e. after §3.1). §3.4 waits on §5.

---

## 4. Teacher/student classification & instructional-time analytics

Our four hand-weighted signals are a good base the published detection work doesn't even attempt. Prior art suggests three additions:

### 4.1 Adopt Moodoo's validated teacher-trajectory feature set

Moodoo (UWB, 2 Hz, 10–21 cm accuracy; 7 teachers / 18 classes) is the most methodologically mature teacher-mobility strand and its features **transfer directly to a tracked video trajectory** (Moodoo, PMC7334189): dwell **"stops"** (≥10 s within a ~1 m radius), transitions (Kalman-smoothed), **Shannon spatial entropy** on a 1 m floor grid, per-zone / per-student-cluster attention time, visit frequency, **Gini index** of spatial dispersion, distance walked, walking speed. These differentiate _supervisory_ vs _focused_ teaching and are pedagogically interpretable — replacing ad-hoc heatmaps with peer-reviewed constructs, and requiring **no engagement claim**.
**Drop-in point:** compute these on the teacher track in `events.py` alongside `spatial_heatmap`/`occupancy_buckets`. Requires §5's metric coordinates for the 1 m / 10 s thresholds to mean anything (see §5). This is the natural home for our circulation-heatmap path — make it first-class and validate thresholds against a lightweight per-room ground-truth pass (ClassID warns behavioural thresholds need per-room calibration; Patidar et al., IMWUT 2024).
**Effort:** medium · **Impact:** high.

### 4.2 Keep the classifier persistent and geometry-aware (we already do — extend it)

Our relative/outlier-margin rule and `teacher_chain` stitching are the right architecture. Extend the signal set with the geometry Moodoo/EduSense rely on: front-of-room zone occupancy, standing height/aspect vs seated students, sustained mobility, being the lone consistently-moving track. This is more robust than any single-frame cue and prevents the id-steal mislabeling. No change needed to the _approach_ — it's ahead of the field here.
**Effort:** low · **Impact:** high.

### 4.3 Instructional-time: use momentary-time-sampling (interval) aggregation

Education researchers quantify on-task behaviour with **momentary time sampling / interval coding**, and Academic Learning Time (ALT) is defined on intervals, not instants (classroom-observation protocol literature). Store per-interval on-task/off-task **counts** rather than per-frame labels — this makes our outputs comparable to human-coded observation and to a validated construct instead of an invented metric. No single behaviour _frame_ equals "on-task."
**Drop-in point:** `events.py::occupancy_buckets` already buckets — align the bucket to a fixed momentary-time-sampling interval and store per-interval counts in the TimescaleDB rollup.
**Effort:** low · **Impact:** medium.

### 4.4 Benchmark the pose-behaviour head externally

Report **mAP@50 / mAP@50:95** against SCB-Dataset3's 6 classes (YOLOv7x baseline 80.3% mAP@50; arXiv:2310.02522) and add a congested-scene test (2026 highly-congested classroom dataset, arXiv:2606.21568). Gives a citable external accuracy figure instead of self-reported quality, and exposes crowding failure modes before deployment.
**Effort:** medium · **Impact:** medium.

**Note on hand-raise / fine pose:** EduSense found off-the-shelf OpenPose unstable in the wild (needed retraining + logical pose filtering; ~92% body detection but much worse on facial/lower-body joints under occlusion). Prefer a **view-invariant, occlusion-robust** hand-raise approach (image-region + temporal cues over raw joints; Springer 2023) and lean on BoT-SORT track continuity to bridge occluded frames rather than emitting per-frame events. Add logical/anatomical + temporal validation on YOLO26-pose keypoints.

---

## 5. Camera homography — add metric calibration (yes)

**Verdict: add a one-time ground-plane homography.** Today `geometry.py` and `events.py::spatial_heatmap` work in normalized pixel space, so proximity/dwell/speed are camera-dependent and non-comparable across rooms — which means Moodoo's thresholds (§4.1) and any cross-lesson claim are unfounded. A single planar homography `H` maps floor pixels → metric floor coordinates for a fixed camera (Galliot; MVA 2022).

### 5.1 The one-time calibration recipe

1. **Undistort first.** Wide-angle CCTV has radial distortion that bends straight floor lines; a homography can only model a straight-line-preserving map. Run OpenCV intrinsic calibration once (`cv2.fisheye.calibrate` if fisheye, else `cv2.calibrateCamera`) → `K`, dist coeffs; undistort every frame **before** inference (OpenCV fisheye docs; arXiv:2001.07243).
2. **Anchor to real scale.** Survey 4+ floor points with known metric coordinates (tape a 1 m square, or use the known floor-tile pitch), click their undistorted pixels, solve with `cv2.getPerspectiveTransform` (4 pts) or `cv2.findHomography(RANSAC)` (more).
3. **Persist** `{K, dist, H, floor_origin, unit=metres}` as a JSON calibration record keyed to the camera/video, mirroring our per-video metadata in TimescaleDB.
4. **Project the foot point.** Use the **mid-ankle keypoints YOLO26-pose already outputs** (mean of L/R ankle, or the lower; fall back to bbox bottom-centre only when ankles are occluded/low-confidence) — more reliable than the jittery bbox bottom our role logic currently keys off (CamLoc, arXiv:1812.11209). Store both pixel foot point and metric `(x_m, y_m)` per detection.
5. **Optional zero-touch fallback** for rooms where taping is impractical: accumulate a few minutes of upright walking tracks, estimate vanishing points from trunk axes + motion, fix scale with an assumed height (children's height priors differ from adult-tuned methods — disclose this as a systematic bias, not a measurement) (Surveillance autocalibration, ECCV 2016).

### 5.2 Ground-plane speed (fixes our id-steal spikes)

Compute speed as metric displacement / dt, **smoothed on the plane** (EMA or a ground-plane Kalman velocity state), and **clamp/flag** speeds above ~2.5 m/s as tracker id-switch artifacts. This turns our pixel-motion circulation metrics into defensible "metres walked / active circulation time" and auto-rejects the id-steal teleport spikes commit `d8489b6` hand-stitched (UCMCTrack; arXiv:2604.10805).

### 5.3 Per-detection uncertainty — feed `quality.py`

Propagate each detection's image-plane noise (scaled by bbox w/h) through `H`'s Jacobian to get a **non-diagonal, distance-dependent ground-plane covariance** `R_k = C·R_uv·Cᵀ` (UCMCTrack's Correlated Measurement Distribution, arXiv:2312.08952). Feed it into association (§3.4) **and** into `quality.py` as an explicit per-row error radius, so the dashboard can widen error bars / grey-out far-field students instead of reporting false-precision centimetres.

### 5.4 Accuracy caveats (must be honest, must be in-product)

- Homography distance error grows **~quadratically** with distance from camera — report accuracy _as a function of distance_, never one flat room-wide figure (arXiv:2604.10805). Near-camera reliable; far-field degrades sharply.
- Valid **only for feet on the calibrated floor plane** — a student on a chair, feet tucked/occluded under a desk, or on a ramp violates the single-plane assumption and `H` cannot detect the error.
- **No vertical/3D** (true height, hand-raise elevation) from a ground-plane homography alone.
- Metric numbers are only as good as identity continuity — id switches produce teleport spikes; plausibility-gate, never report raw.
- UCMCTrack tolerates modest calibration error but degrades past ~10° tilt/pan error — a useful tolerance for how carefully to estimate extrinsics.
- A homography fit on a **non-undistorted** wide-angle frame is invalid at the edges where students often sit — don't skip undistortion and still claim peripheral accuracy.

**Effort:** medium (calibration + foot projection) → high (full CMD covariance) · **Impact:** high.

---

## 6. Ethics & honesty guardrails — CAN / CANNOT claim

Three independent bodies of evidence (scientific validity, EU AI Act, US FERPA/BIPA) converge on one boundary: **claim behaviour and geometry; never claim inner states.** Emotion/affect is not validly inferable from faces or bodies (Barrett et al. 2019, _Psychological Science in the Public Interest_), face-derived scoring fails hardest on darker-skinned and female subjects (Gender Shades: <1% vs 34.7% error; Lockport 16× worse for Black women), and inferring students' attention/motivation via biometrics is **prohibited** in EU education (AI Act Art. 5(1)(f), enforced from 2 Feb 2025). Pose skeletons and re-ID embeddings can themselves be **behavioural biometric data** (AI Act Art. 3(34)) — de-identification must be _enforced by aggregation + k-anonymity + short retention_, not assumed from the absence of a face crop.

| We CAN claim (defensible)                                                                                            | We CANNOT claim (prohibited / invalid)                                                                                                                                                 |
| -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Time-on-task via **interval/momentary-time-sampling** counts                                                         | Student **"engagement"** as a validated construct — every dataset (DAiSEE, EngageNet, DIPSER) labels it via crowd/self/expert proxies, none vs learning outcomes                       |
| In-seat vs out-of-seat; presence / head-count                                                                        | **"Attention," "focus," "motivation," "interest"** as mental states — inferring attention/motivation via biometrics is named-prohibited in EU education (Art. 5(1)(f))                 |
| Hand-raise **counts** (view-invariant, occlusion-robust, temporally smoothed)                                        | **Emotion, affect, mood, boredom, confusion, arousal** from face/body — no validated mapping (Barrett 2019); banned in EU education                                                    |
| Gross movement / **teacher circulation zones**, dwell "stops", metres walked                                         | That head/torso orientation is **gaze or attention** — it's an "orientation-toward-board" proxy only; true gaze isn't recoverable at classroom distances (GESCAM, CVPRW 2024; ClassID) |
| **Posture as physical configuration** (standing/seated) with detector confidence attached                            | That any single behaviour **frame** = "on-task" — requires interval aggregation                                                                                                        |
| Teacher proximity / dwell / coverage in **metres** (post-homography), with distance-dependent error bars             | **Pixel** proximity/dwell as comparable across cameras/rooms without a metric floor-plan homography                                                                                    |
| Detection/tracking accuracy (mAP, HOTA, quality tiers)                                                               | That analytics are **demographically fair/unbiased** for per-individual scoring — no subgroup validation; face pipelines documented to fail worst on darker-skinned/female subjects    |
| **Aggregate, zone-level, teacher-facing reflection** (participation balance, circulation heatmap, hand-raise trends) | Outputs suitable for **grading, discipline, teacher evaluation, admissions**, or any high-stakes individual decision                                                                   |
| That data is de-identified **because** of aggregation + k-anonymity + non-persistence                                | That data is "anonymous" merely because faces are blurred / only skeletons stored — pose/gait/re-ID embeddings can be biometric                                                        |
| Within-session track continuity via **ephemeral** re-ID                                                              | **Per-student longitudinal profiles / "progress"** — reintroduces identifiable records and the exact profiling regulators fined schools for                                            |

**Enforcing guardrails on our stack:**

- **Rename + delete** `emotion/affect/engagement/attention/motivation/mood` from schema (`models.py`: `EventOut`, `AnalyticsOut`, `HeatmapOut`), UI, DB columns and API. Low effort, high impact — a rule-based classifier that emits an "engagement score" walks straight into Art. 5(1)(f) and the Barrett invalidity critique.
- **Ephemeral, non-galleried re-ID.** Use CLIP-ReID/SOLIDER embeddings only as within-session track-linking keys — in-memory, salt+rotate per session, **never persist a durable gallery, never map to a name/roster**. Persisting face-derived embeddings is the exact trigger for GDPR Art. 9 / BIPA/CUBI (Meta paid $650M BIPA, $1.4B Texas CUBI) and the NY school FRT ban. ClassID shows pose/appearance re-ID gives continuity without face storage; aggregate identities to **seat/zone slots**.
- **Aggregate before persisting.** Write class/zone-level counts to TimescaleDB; enforce a **k-anonymity threshold** (suppress buckets < k, e.g. k=5); store no per-pupil time series by default. Per-student rows make the DB an FERPA education record with access/consent duties.
- **Edge processing + short retention TTL.** Discard raw frames immediately (minutes-scale TTL), persist only derived aggregates, and document a posted retention/destruction schedule (BIPA/CUBI require one; Sweden's first GDPR fine hinged on necessity/minimisation — Skellefteå school, SEK 200,000).
- **Product framing = private teacher reflection, not per-student surveillance/scoring.** ClassInSight/ClassMind show reflection framing is what teachers adopt; Hikvision's per-student scoring produced "performative personalities" and backlash. **Remove any per-student real-time score, leaderboard, or ranking view.**
- **Ship a DPIA + versioned "can-claim/cannot-claim" statement + bias/validity disclaimer.** Consent is _insufficient_ in schools (power imbalance, per the Swedish DPA) — rely on data-minimised necessity + notice. Geo-gate affect features off in the EU and, given convergence, off everywhere.

---

## 7. Ranked roadmap

Ship-order chosen so each item unblocks the next: metric coordinates (item 3) unlock validated teacher analytics (item 6) and ground-plane association; discriminative embeddings (item 2) unlock GTA (item 5).

| #   | Item                                                                                                           | Area              | Effort  | Impact | Ship   | Prior-art justification                                                                                     |
| --- | -------------------------------------------------------------------------------------------------------------- | ----------------- | ------- | ------ | ------ | ----------------------------------------------------------------------------------------------------------- |
| 1   | Rename/delete affect metrics; ephemeral non-galleried re-ID; k-anon aggregate persistence; retention TTL; DPIA | Ethics/legal      | Low–Med | High   | **1**  | AI Act Art. 5(1)(f), Barrett 2019, GDPR Art. 9, BIPA/CUBI, Swedish DPA fine — non-negotiable safe harbor    |
| 2   | CLIP-ReID/SOLIDER backbone (TensorRT FP16) replacing CLIP-general+HSV, fine-tuned per room                     | Track/re-ID       | Med     | High   | **2**  | MSMT17 mAP 73–77 vs OSNet 52.9; CLIP-ReID text-label-free in-domain training (arXiv:2211.13977, 2303.17602) |
| 3   | Ground-plane homography: undistort → survey 4+ pts → `H`; project mid-ankle foot point; store `{K,dist,H}`     | Homography        | Med     | High   | **3**  | Pixel metrics non-comparable; foot-point localization (Galliot, CamLoc, UCMCTrack)                          |
| 4   | OC-SORT ORU/OCM + Hybrid-SORT weak cues into BoT-SORT association                                              | Track/re-ID       | Med     | High   | **4**  | DanceTrack 53.8→61–66 HOTA, appearance-free (OC-SORT CVPR23, Hybrid-SORT arXiv:2308.00783)                  |
| 5   | GTA-style offline global tracklet association replacing greedy heap merge                                      | Track/re-ID       | High    | High   | **5**  | +10.24 HOTA (SORT), 81.04 HOTA SportsMOT, temporal-conflict constraint (arXiv:2411.08216)                   |
| 6   | Moodoo teacher-trajectory features (stops, entropy, Gini, coverage) in metres                                  | Teacher analytics | Med     | High   | **6**  | Validated supervisory-vs-focused constructs (Moodoo, PMC7334189) — needs items 1,3                          |
| 7   | Ground-plane speed smoothing + plausibility clamp; per-detection CMD covariance → `quality.py`                 | Homography/track  | Med     | High   | **7**  | Kills id-steal teleport spikes; honest error bars (UCMCTrack, arXiv:2604.10805)                             |
| 8   | Momentary-time-sampling interval aggregation for on-task rollups                                               | Instr.-time       | Low     | Med    | **8**  | ALT / classroom-observation protocol validity                                                               |
| 9   | BoostTrack++ trio + Deep-EIoU low-effort association drop-ins                                                  | Track/re-ID       | Low     | Med    | **9**  | MOT-agnostic, revives occluded detections (arXiv:2408.13003, 2602.00484)                                    |
| 10  | View-invariant occlusion-robust hand-raise + logical pose filtering                                            | Behaviour         | Med     | Med    | **10** | EduSense OpenPose instability; Springer 2023 view-invariant approach                                        |
| 11  | UCMCTrack ground-plane Mahalanobis association                                                                 | Track/re-ID       | Med     | Med    | **11** | 63.6 HOTA motion-only; needs item 3 (arXiv:2312.08952)                                                      |
| 12  | SCB-Dataset3 / PTPD + congested-scene benchmark of the pose-behaviour head                                     | Validation        | Med     | Med    | **12** | External citable mAP baseline (arXiv:2310.02522, 2606.21568)                                                |

---

### Sources (representative)

**Classroom analytics & engagement:** SCB-Dataset3 (arXiv:2310.02522); AlphaPose+Faster R-CNN teacher (PMC12193412); Moodoo (PMC7334189); DAiSEE (arXiv:1609.01885); EngageNet (arXiv:2302.00431); DIPSER (arXiv:2502.20209); Booth et al. engagement tutorial (CU Boulder 2023); GESCAM (CVPRW 2024); EduSense (IMWUT 2019); ClassID (IMWUT 2024); ClassInSight (CHI 2024); ClassMind (arXiv:2509.18020); Hikvision/Hangzhou No. 11 (Sixth Tone).
**Tracking & re-ID:** ByteTrack (arXiv:2110.06864); OC-SORT / Deep-OC-SORT (arXiv:2302.11813); Hybrid-SORT (arXiv:2308.00783); UCMCTrack (arXiv:2312.08952); BoostTrack++ (arXiv:2408.13003); GTA (arXiv:2411.08216); Deep-EIoU/GTATrack (arXiv:2602.00484); CLIP-ReID (arXiv:2211.13977); TransReID (arXiv:2102.04378); SOLIDER (arXiv:2303.17602); MOT survey (arXiv:2506.13457).
**Homography/calibration:** UCMCTrack (arXiv:2312.08952); CamLoc (arXiv:1812.11209); homography distance-error modeling (arXiv:2604.10805); OpenCV fisheye docs; Criminisi/Reid/Zisserman Single View Metrology (IJCV 2000); Galliot homography calibration.
**Ethics/law:** Barrett et al. 2019 (PSPI); Gender Shades (PMLR v81 2018); EU AI Act Art. 5(1)(f)/3(34)/3(39) + FPF Red Lines analysis; GDPR Art. 9 + Swedish DPA (IMY 2019); FERPA education-record/biometric FAQs; BIPA/CUBI (Meta $650M / $1.4B); NY school FRT ban (2023).
