# ClassroomCV — Architecture & Product Decision Doc

_Synthesis of six research lenses (detection/pose, tracking/re-ID, streaming architecture, TimescaleDB, product strategy, branding) into one prioritized plan._

**Status:** proposed · **Owner:** lead architect · **Scope:** the ml-service detection/track pipeline, the TimescaleDB event store, the live-streaming path, and the owner-facing product.

---

## 1. Executive summary

Do **not** chase a bigger model. The highest-leverage moves are all cheap and shippable on the current Apple-Silicon stack: swap **YOLO11m-pose → YOLO26m-pose** (CoreML fp16, near drop-in, +AP on small objects) and add **SAHI tiling on a back-rows ROI**; make tracking **appearance-light** by disabling GMC on our static cameras, moving association to a **ground-plane Kalman + Mahalanobis cost (UCMCTrack-style)**, adding **OC-SORT re-update/momentum**, and replacing greedy identity chaining with a **sliding-window global Hungarian** — treating the teacher as a **dedicated single-target re-ID gallery**. For scale, keep the existing `iter_frames` seam as a frozen **FrameSource contract**, evolve the DB writer from whole-video `replace_detections` to **append-only, idempotent, offset-keyed windowed upserts**, and shard 80 cameras across a **Kafka consumer group keyed by camera_id**. Retype `detection_events` to typed `int2` columns, set **~1h chunks**, enable the **Hypercore columnstore (segmentby `video_id`)**, and build the occupancy cagg with **HyperLogLog** (not `count(distinct)`). The new _product_ value is mostly **re-derivation, not new ML**: instructional-time composition, coverage/unsupervised-time alerting, room utilization, punctuality, and a longitudinal benchmarking layer — sold under an explicit **no-face-recognition, aggregate-only** posture. **The one thing that requires leaving Apple Silicon:** the live 80-camera fleet needs a 2–4 GPU NVIDIA tier (batched NVDEC + TensorRT); MPS stays for dev and single-video VOD.

---

## 2. Detection & pose — verdict

**Keep the YOLO/Ultralytics family. Switch YOLO11m-pose → YOLO26m-pose. Do NOT adopt YOLOv12.**

**Single highest-impact change:** benchmark **YOLO26m-pose exported to CoreML fp16** as a drop-in for YOLO11m-pose. It reports up to +7.2 AP over YOLO11 on COCO pose, is NMS-free/end-to-end, uses RLE keypoints, and its STAL training recipe specifically improves small-object label coverage — which is exactly our failure mode (40–120 px occluded back-row students). Same Ultralytics API, so BoT-SORT / CLIP-ReID / SAM2 plumbing is unchanged. Validate accuracy on a labeled classroom clip and measure CoreML fp16 latency at imgsz 1280–1536 before committing.

**Hard no: YOLOv12 / YOLOv12-pose on this hardware.** Its accuracy depends on CUDA-only FlashAttention; on MPS/ANE it runs _slower_ than YOLO11 with no benefit. Spending effort here regresses latency at fleet scale.

**Second lever (medium effort, high impact): SAHI-style tiling on a cropped back-rows ROI**, not the whole frame. Tiling recovered small-object recall 31.8% → 86.4% in a surveillance case (+5–7 AP inference-only, +12–14 AP with slice-aware fine-tuning). Restrict tiles to the upper/back band of the 2560×1440 frame to bound the inference multiplier, and merge with NMM/Greedy-NMM, not plain NMS.

**Cheaper small-object levers (low effort, do alongside):** raise pose `imgsz` toward native (1440/1536) — running at 1280 halves a 2560-wide frame before the smallest students are ever seen — and/or fine-tune a **P2 (stride-4) head**. Both are YOLO11/26-compatible and dodge the DETR-on-Mac penalties.

**Architecture guardrail:** keep the always-on realtime path **one-stage/bottom-up** (YOLO-pose / RTMO), whose cost is fixed per frame. Top-down transformers (ViTPose++, Sapiens) scale with people — prohibitive at 80 cams × ~30 students live — so **reserve them for offline auto-labeling / pseudo-ground-truth** to fine-tune and evaluate the realtime model.

**Robustness & calibration (medium/low effort):** fine-tune on classroom data with corruption-style augmentation (motion blur from the moving teacher, exposure/gamma, noise, IR/night), and **post-hoc temperature-calibrate** detection/keypoint confidence on a held-out classroom set — calibrated scores stabilize presence thresholds, teacher/student gating, and ReID accept/reject, and directly attack the id-steal bugs already logged in this repo.

**Do not adopt now:** RT-DETR / D-FINE / DEIMv2 (box-only, gains skew to large objects, CoreML/ANE op-fallbacks kill their Mac speed). If detection ever leaves Ultralytics, prefer **RTMO / DETRPose** over RT-DETR for this crowd+pose+Mac combination — but treat their Apple-Silicon export as unproven.

---

## 3. Tracking & re-ID for identical uniforms — ranked upgrade path

Governing evidence: **DanceTrack** (uniform appearance) proves the bottleneck is **motion + location association, not appearance or ReID backbone quality**. HOTA climbs ByteTrack ~47.7 → OC-SORT ~55 → Deep-OC-SORT ~61 → Hybrid-SORT ~65.7 from occlusion-robust motion, not stronger appearance. Rank the work accordingly:

1. **Disable GMC on the static cameras (near-zero effort, high impact).** BoT-SORT's ECC global-motion-compensation only injects Kalman noise on a fixed camera. Turn it off.
2. **Add OC-SORT's Observation-Centric Re-Update (ORU) + Momentum (OCM) on the existing Kalman (low effort, high impact).** ORU rebuilds a virtual trajectory through a desk occlusion; OCM enforces velocity-direction consistency so crossing students don't swap IDs. Drop-in, no ReID needed.
3. **Move association onto a ground-plane Kalman with Mapped Mahalanobis cost, UCMCTrack-style (medium effort, high impact).** One static homography per camera; appearance-free, CPU-cheap (>1000 FPS), leverages fixed-camera + near-desk geometry. This is the correct "motion compensation" for our cameras.
4. **Replace greedy identity chaining + seat-anchor merge with a sliding-window GLOBAL Hungarian tracklet association (medium effort, high impact).** Cost = ground-plane distance + seat prior + **low-weight, gated** ReID. This is the SOTA fix for multi-minute occlusion and the returning teacher; fold the seat-anchor merge in as a cost term, not a separate greedy step.
5. **Model the teacher as a dedicated single-target re-ID gallery (medium effort, high impact).** She is visually distinct and _must_ survive occlusion — a KeepTrack-style maintained embedding gallery (ReID median + variance) + motion gate, re-acquired by appearance AND trajectory, beats hoping the crowd tracker re-merges her. Keep it **separate from the student pool.**
6. **Add Hybrid-SORT weak cues — bbox height + detection-confidence modeling — into the cost (low effort, medium impact).** Height disambiguates a standing teacher from seated students; near free.
7. **Formalize the seat/location prior as a per-seat occupancy Gaussian in the cost (low effort, medium impact).** Turns the seat-anchor heuristic into a principled always-on term that also feeds the global association.
8. **Swap general CLIP ViT-B/32 + HSV histogram → a dedicated ReID backbone (CLIP-ReID or SOLIDER), but down-weight and gate it (medium effort, medium impact).** These extract more of the residual signal (teacher vs student, hair/skin, uniform wear) but inter-student discrimination on identical uniforms is near-random — use ReID only as a motion-ambiguity tie-breaker and in global re-linking, never as the frame-to-frame driver.
9. **Stop investing in pose-head ReID features (low impact).** Not identity-trained; spend the budget on motion/geometry and global association.

---

## 4. Streaming architecture for 80 cams × 8h — staged plan on the existing seam

The `detector.iter_frames` generator already yields `(video_ts_ms, frame)` and owns the grab/retrieve/stride loop — it is a correctly-placed **FrameSource** seam. The plan is the industry **Kappa** pattern: one detection core, pluggable sources, windows just get shorter for live. **Evolve the writer before the reader**, and prove each stage against the current test suite with a file-replay producer ("simulated live") before real cameras exist.

**Stage 1 — Freeze the FrameSource contract (low effort, high impact).** Define an interface implemented by `FileFrameSource`, `RtspFrameSource`, `KafkaFrameSource`, all yielding `(ts_ms, frame, camera_id, offset)`. `detect_video` stays byte-identical. Unblocks everything with zero behavioral risk.

**Stage 2 — Append-only, idempotent, windowed writer (high effort, high impact — the biggest change).** `db.replace_detections` (whole-video DELETE + COPY, atomic) cannot serve an unbounded 8h stream. Introduce `append_window(camera_id, window_start, rows, offset)` that inserts finalized-window detections **and** upserts the `(topic, partition, offset)` high-water mark in the **same TimescaleDB transaction** → at-least-once ingest with exactly-once effects (a Kafka redelivery re-runs the window but the offset guard makes it a no-op). Reuse the existing `run_tokens`/`StaleRunError` fence per-window. **Keep `replace_detections` unchanged for VOD.**

**Stage 3 — Online, bounded tracker + gallery re-ID (high effort, high impact).** `remerge_from_raw` + whole-video CLIP merge assume the full video is in memory. Adopt DeepStream's proven shape: a bounded per-track gallery (~last 100 embeddings), match new tracks by cosine-NN within a sliding window, finalize identities on a 30–60 s lag. The teacher re-merge becomes "gallery hit across a gap." Memory stays O(active tracks) per camera so a camera's whole state lives on one worker.

**Stage 4 — Shard cameras across a GPU-worker fleet via a Kafka consumer group keyed on camera_id (medium effort, high impact).** One camera → one partition → one worker (tracking is stateful; **never split a camera mid-stream**). ~20–40 cameras/worker on NVIDIA. Rebalancing on worker loss reassigns whole cameras (gallery cold-starts). This replaces the single daemon worker thread in `jobs.py` for the live path.

**Stage 5 — NVIDIA fleet + GStreamer/NVDEC decode (medium effort, high impact) — ⚠️ REQUIRES LEAVING APPLE SILICON.** 80 cams × 5 fps × 1280px YOLO ≈ 400 inf/s. MPS has no `nvstreammux` batching and no batched NVDEC; this needs batched TensorRT on **2–4 L4/A10-class GPUs**. Move the live decode to GStreamer (`uridecodebin`/`nvv4l2decoder`) behind the frozen FrameSource. **Apple Silicon stays for dev and the single-video VOD worker** (it wins ~5–10× on watts/inference but loses multiples on aggregate throughput). `device` is a config knob (`mps`|`cuda`) so the core stays portable.

**Backpressure:** live wants **drop-to-latest** — a small per-camera ring buffer that overwrites oldest frames, plus Kafka partition-pause (commit offsets only after a window is persisted) when the worker's inflight window saturates. VOD keeps its current zero-drop bounded queue (maxsize=4). Make drop policy a per-source setting.

**Build vs buy:** a home-grown Python/Kafka pipeline is correct at 80 cameras (2–4 GPUs, one codebase). Consider a full **DeepStream** rewrite only past **~150 cameras**, where `nvstreammux` batching and NvMultiObjectTracker's 128-stream batches dominate.

---

## 5. TimescaleDB at scale — concrete DDL & policy changes

Volume: 80 cams × 8h × 5 fps × ~30 people ≈ **345.6M rows/day** (~12k rows/s in class hours). The schema decisions below are the difference between workable and unqueryable.

**5.1 Retype `detection_events` to fully typed columns (medium effort, high impact — do first).** Replace jsonb bbox with `int2` `x,y,w,h` and `int2` scaled confidence (0–1000). Typed columns compress ~3× better (delta/Gorilla-encodable, ~1.5–2 GB/day vs ~5–7 GB/day jsonb) and skip per-row JSON parsing on the hot dashboard read. Normalized floats → `round(x*10000)` (fits `int2`); 4K pixel coords already fit `int2`.

**5.2 Set `chunk_time_interval` to ~1 hour — never leave the 7-day default (low effort, high impact).** A 1h chunk ≈ 43M rows / ~3.5 GB, keeping the active uncompressed chunk within ~25% of RAM. The 7-day default builds ~200 GB chunks that cannot be indexed/vacuumed in RAM. Verify with `chunks_detailed_size`; drop to 6h/30m by RAM; don't go below a few hundred MB/chunk (wrecks compression ratios).

**5.3 Enable the Hypercore columnstore (2.18+ API) (low effort, high impact):**

```sql
ALTER TABLE detection_events SET (
  timescaledb.enable_columnstore = true,
  timescaledb.segmentby = 'video_id',
  timescaledb.orderby   = 'video_ts_ms'
);
SELECT add_columnstore_policy('detection_events', after => INTERVAL '7 days');
```

`segmentby video_id` gives per-video segment pruning for both dashboard reads (`WHERE video_id=$1`) and `/rederive` DELETEs; setting the policy lag (7d) **larger than the reprocessing SLA** guarantees re-derivation lands on rowstore and never triggers decompress-modify-recompress. **Do not** add `track_no` to `segmentby` (high-cardinality → tiny batches → poor ratio). **Do not** `add_dimension` space-partition on a single node (discouraged; targets deprecated multi-node) — model per-school via a `school_id` column + leading composite index instead.

**5.4 Occupancy cagg with HyperLogLog, not `count(distinct)` (medium effort, high impact).** `count(distinct track_no)` is rejected inside a continuous aggregate. Use toolkit `hyperloglog(track_no)` (or `approx_count_distinct`, ~2% error, mergeable) for "unique people per minute" and read via `distinct_count(rollup(hll))`. **"Max concurrent people in frame" is NOT a distinct-count** — materialize the per-frame `count(*)` the ML layer already computes as occupancy points, then `max`/`avg` over it.

**5.5 Composite index + precomputed playback (medium effort, high impact).** `getDetections`' `count(distinct video_ts_ms)` + `row_number()` stride is two full scans of up to 4.3M rows/video on every load. Add a btree index on `(video_id, video_ts_ms)` and precompute downsampled playback frames (a cagg or an RDP-simplified overlay table) instead of deriving stride at request time.

**5.6 Hierarchical caggs 1m → 15m → 1h via `rollup()` (medium effort, medium impact).** Integer-multiple buckets; `rollup()` merges HLL and numeric partial states. Keep `materialized_only=true` for finished-class historical views (faster); only the live "class in progress" view needs `materialized_only=false` (pays a raw-tail scan for freshness — off by default since 2.13).

**5.7 Three-tier lifecycle (medium effort, medium impact).** `add_retention_policy` drops raw detection chunks after 30–90d; caggs kept forever; plus an app-level **RDP-simplified per-track polyline overlay table** (plain, non-hyper) as the durable visual tier that survives raw retention and drives heatmap/path replay cheaply. (Cloud-only data-tiering to S3/Parquet is unavailable on our self-hosted deployment — the RDP overlay + cagg tier is the substitute.)

---

## 6. School-owner insights — ranked NEW analytics, and what NOT to claim

The pipeline already persists per detection: timestamp, bbox pos+size, track id, role, confidence, `standing`, `back_to_camera`, plus transient pose keypoints. **Most high-value metrics are re-derivations over data already in TimescaleDB, not new ML.**

**Build, in priority order:**

1. **Instructional-Time Composition (low effort, high impact).** Board / circulating-among-desks / front-of-room / absent, framed on Academic Learning Time research. Reuses `presence_intervals`, `board_intervals`, the circulating split, and the teacher heatmap. Answers the owner's #1 question — "is class time used to teach?"
2. **Coverage / unsupervised-time alerting — substitute verification (low effort, high impact).** Fire an alert when no teacher-role person is present beyond a threshold during a scheduled block. High liability/safety value. Sell honestly as "a teacher-role adult was present X% of the block" — **never** identity verification (keeps us clear of biometric law).
3. **Room Utilization vs capacity (low effort, high impact).** Reframe existing occupancy buckets against a room-capacity/schedule input; idle-period + peak-concurrency detection. Mirrors the proven higher-ed space-utilization market (~45% average utilization). Lowest privacy risk — counts rooms, not children.
4. **Longitudinal + cross-classroom benchmarking layer (medium effort, high impact).** Day/week/term deltas, formative percentiles, weekly PDF/email. The individual-video metrics are features; **trends and comparisons are the leadership product.** Frame cross-teacher comparison as formative/coaching with k-anonymity — never punitive ranking.
5. **Punctuality + settle-time (medium effort, medium impact).** On-time start/end and time-to-settle from the seated-fraction curve (derived from the `standing` flag) + a schedule input. "Dead time" is owner-legible lost minutes.
6. **Teacher Mobility / Circulation index (medium effort, medium impact).** Coverage/dwell-entropy + "did she reach the back rows" from the existing heatmap. Label it **image-plane** coverage; per-student proximity-in-feet only _after_ camera homography.
7. **Aggregate hand-raise participation proxy (medium effort, medium impact).** Wrist-above-shoulder from existing keypoints → class-level participation-opportunity count. The one genuinely new engagement-adjacent signal — **only** if kept aggregate, k-anonymized, camera-angle-caveated, never per-child.

**Do NOT claim (technically indefensible and/or an ethics/legal landmine):**

- **Student engagement / attention / emotion / learning / lesson quality scores** — affect-from-video is documented as invalid and biased; we have no audio and no comprehension signal.
- **Per-named-student attendance** — occupancy is a headcount _proxy_ (track fragmentation over-counts, occlusion under-counts); label it "occupancy trend," not "attendance." No face recognition, by design.
- **Talk-time / teacher-vs-student discussion** — requires audio we don't ingest.
- **Proximity in feet** — all positions are image-plane; needs homography first.

**Privacy posture as the lead differentiator (medium effort, high impact).** Amid the edtech-surveillance backlash, "no facial recognition, no named students, aggregate-only, short raw-video retention, ephemeral (never persisted) re-ID embeddings" is the top sales differentiator, not just an FERPA/GDPR/BIPA floor. Concretely: keep CLIP-ReID/student embeddings **ephemeral / in-video only**, publish a retention policy, derive metrics then short-retain/discard raw frames, run a DPIA for EU/UK sites, add k-anonymity suppression for low-headcount rooms, and separate teacher-facing (coaching) from owner-facing (operations) views. **Language guardrail:** say "insight / clarity / presence / see," never "detect / monitor / track / surveil."

---

## 7. Branding

**Product name: Luminary.** It is the only candidate that hits every constraint at once — light/clarity (a celestial luminary), pedagogy (a luminary is an inspiring, leading teacher), premium/aspirational tone, and, decisively, an **anti-surveillance frame**: the product _elevates_ teachers into luminaries rather than monitoring them.

**Tagline:** _"Every lesson, brought to light."_ (puns on the name; carries both "reveal understanding" and "the opposite of covert observation"). A/B alternates: _"Where good teaching comes to light."_ / _"Clarity for every classroom."_

**Immediate action:** formal trademark clearing in the education-software class (the one collision to check is the consumer **podcast** app "Luminary" — a different Nice class); secure `luminary.education` + `getluminary`/`heyluminary`; keep "Luminary for Schools" as a defensive lockup.

**Logo concept:** a warm-amber "light" dot floating just above a chalk-white capsule stroke, set in a spruce-ink squircle — reads as a candle/beam of learning and an upright figure at the board topped by a point of light. Resolves at 16px to a chalk bar + one bright amber dot. **Explicitly avoid** any eye/iris/aperture/lens/camera shape, the four-point AI sparkle, and the magnifying glass. Add a soft amber radial glow behind the dot at large sizes; drop the glow below 32px.

**Color — "Chalkboard & Warm Light" (OKLCH):** ink `oklch(0.27 0.03 165)`, primary `oklch(0.50 0.085 168)`, accent "light" (≤5% use) `oklch(0.80 0.135 72)`, surface a **cool** chalk-white `oklch(0.985 0.006 150)` (deliberately NOT the AI-default cream/sand). Danger is a muted brick `oklch(0.55 0.16 25)`, not fire-red. By construction this is none of the three category clichés: cream/sand AI default, navy-and-gold fintech, or SaaS purple gradient.

**Type:** **Fraunces** (variable serif, low softness) for wordmark + hero/section headings; **Hanken Grotesk** for all UI/body with **tabular lining figures** on every metric and axis; **Geist Mono** for timecodes, track IDs, and data chips. Editorial-intelligence-meets-precise-instrument, escaping both all-Inter AI-slop and the fintech Didone look.

**Motion:** calm ease-out `cubic-bezier(0.22, 1, 0.36, 1)` at 180–240ms; one signature non-looping amber "bring-to-light" bloom on dashboard load; honor `prefers-reduced-motion` (opacity-only).

---

## 8. Prioritized roadmap

Ship-order groups: **P0** = cheap, high-impact, on current stack, unblock others; **P1** = high-impact, medium cost; **P2** = larger / requires new infra; **P3** = product/brand parallel track.

| #   | Item                                                                          | Area               | Effort | Impact | Ship-order |
| --- | ----------------------------------------------------------------------------- | ------------------ | ------ | ------ | ---------- |
| 1   | Disable BoT-SORT GMC on static cameras                                        | Tracking           | Low    | High   | P0         |
| 2   | Add OC-SORT ORU + OCM on existing Kalman                                      | Tracking           | Low    | High   | P0         |
| 3   | `chunk_time_interval` → ~1h (verify empirically)                              | TimescaleDB        | Low    | High   | P0         |
| 4   | Enable Hypercore columnstore, segmentby `video_id`, policy after 7d           | TimescaleDB        | Low    | High   | P0         |
| 5   | Composite index `(video_id, video_ts_ms)`                                     | TimescaleDB        | Low    | High   | P0         |
| 6   | Freeze FrameSource contract; add Kafka/RTSP sources + file-replay producer    | Streaming          | Low    | High   | P0         |
| 7   | Benchmark YOLO26m-pose (CoreML fp16) vs YOLO11m-pose on classroom clip        | Detection          | Low    | High   | P0         |
| 8   | Retype `detection_events` to int2 bbox + scaled conf                          | TimescaleDB        | Med    | High   | P0         |
| 9   | Instructional-Time Composition KPI                                            | Product            | Low    | High   | P0         |
| 10  | Coverage / unsupervised-time alerting (substitute verification)               | Product            | Low    | High   | P0         |
| 11  | Room Utilization vs capacity                                                  | Product            | Low    | High   | P0         |
| 12  | Lead privacy posture (ephemeral embeddings, retention, k-anon)                | Product            | Med    | High   | P0         |
| 13  | Post-hoc confidence calibration (temperature scaling)                         | Detection          | Low    | Med    | P1         |
| 14  | Hybrid-SORT weak cues (bbox height + conf) into cost                          | Tracking           | Low    | Med    | P1         |
| 15  | Seat prior as per-seat occupancy Gaussian in cost                             | Tracking           | Low    | Med    | P1         |
| 16  | SAHI tiling on back-rows ROI (NMM merge)                                      | Detection          | Med    | High   | P1         |
| 17  | Raise pose imgsz to 1440/1536 and/or P2 head fine-tune                        | Detection          | Low    | Med    | P1         |
| 18  | Occupancy cagg via HyperLogLog + per-frame count materialization              | TimescaleDB        | Med    | High   | P1         |
| 19  | Ground-plane Kalman + Mahalanobis cost (UCMCTrack-style)                      | Tracking           | Med    | High   | P1         |
| 20  | Teacher single-target re-ID gallery (separate from students)                  | Tracking           | Med    | High   | P1         |
| 21  | Swap CLIP ViT-B/32+HSV → CLIP-ReID/SOLIDER (gated, low-weight)                | Tracking           | Med    | Med    | P1         |
| 22  | Classroom fine-tune with corruption/IR augmentation                           | Detection          | Med    | High   | P1         |
| 23  | Append-only idempotent windowed writer (offset-in-DB)                         | Streaming          | High   | High   | P1         |
| 24  | Online bounded tracker + gallery re-ID (sliding window)                       | Streaming/Tracking | High   | High   | P1         |
| 25  | Sliding-window global Hungarian tracklet association                          | Tracking           | Med    | High   | P1         |
| 26  | Hierarchical caggs 1m→15m→1h; RDP overlay + retention tiers                   | TimescaleDB        | Med    | Med    | P1         |
| 27  | Longitudinal + cross-classroom benchmarking layer                             | Product            | Med    | High   | P1         |
| 28  | Punctuality + settle-time metrics                                             | Product            | Med    | Med    | P2         |
| 29  | Teacher Mobility / Circulation index                                          | Product            | Med    | Med    | P2         |
| 30  | Aggregate hand-raise participation proxy (k-anon)                             | Product            | Med    | Med    | P2         |
| 31  | Shard 80 cams via Kafka consumer group keyed on camera_id                     | Streaming          | Med    | High   | P2         |
| 32  | **NVIDIA GPU fleet (2–4 GPUs) + GStreamer/NVDEC decode** ⚠️ off Apple Silicon | Streaming/Infra    | Med    | High   | P2         |
| 33  | Live drop-to-latest backpressure (per-camera ring + partition pause)          | Streaming          | Med    | Med    | P2         |
| 34  | Offline ViTPose++/Sapiens auto-labeling for fine-tune/eval                    | Detection          | Med    | High   | P2         |
| 35  | Brand rollout: name, logo, OKLCH palette, type, tagline                       | Branding           | Med    | High   | P3         |

**⚠️ Apple-Silicon exit flag:** item **32** is the only one that requires leaving Apple Silicon. Everything through P1 (and most of P2) ships on the current MPS stack; MPS remains the dev + single-video VOD runtime permanently. The live 80-camera fleet is a 2–4 GPU NVIDIA problem because MPS has no `nvstreammux` batching and no batched NVDEC. Evaluate **MLX** (reported 1.1–2.6× over PyTorch-MPS) only if CoreML fp16 latency becomes the VOD/dev bottleneck — it does not substitute for the NVIDIA live tier.
