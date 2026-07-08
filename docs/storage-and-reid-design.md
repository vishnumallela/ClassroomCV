# Luminary — Storage, Re-ID, and Zone-Detection System Design

**Status:** proposed · **Owner:** lead architect · **Scope:** `services/ml-service` + TimescaleDB (`apps/api-service` schema)

This document synthesizes three system-design research lenses (data infrastructure,
production person re-ID, open-vocab zone detection) into a concrete target design for our
exact stack: a TimescaleDB `detection_events` hypertable (jsonb bbox+meta, wall-clock `ts`,
1h chunks, compress after 1h, retain 2 days) + a permanent overlay tier (RDP polyline + 2s
keyframes in `tracks.meta`) + typed `video_analytics`/`tracks`/`events` + a `track_minute`
continuous aggregate; ML = YOLO26-pose + BoT-SORT (gmc off) + CLIP ViT-B/32 median embedding

- HSV torso histogram + hard seat-anchor veto + greedy heap merge over the whole VOD; zones
  via YOLO-World (`yolov8s-worldv2`) + SAM 2 + geometric scoring; deployed on an NVIDIA GPU.

---

## 1. Executive summary — highest-leverage changes

1. **Swap the re-ID backbone: CLIP ViT-B/32 → a re-ID-fine-tuned encoder (OSNet-AIN x1.0 or
   CLIP-ReID ViT-B/16), exported to TensorRT.** This is the single biggest correctness lever.
   CLIP ViT-B/32 is a scene/text-aligned encoder that was never trained to separate people, so
   its same-vs-different-person cosine margin is small — which is exactly why `merge.py` needs a
   low `EMBED_VETO_COS = 0.35` plus heavy HSV/spatial scaffolding to compensate. Every production
   re-ID stack (FastReID, DeepStream ReIdentificationNet, the AICity-2024 winners) routes the
   backbone through BNNeck + L2 before a metric loss; CLIP-ReID exists _because_ raw CLIP
   underperforms until fine-tuned (73–76% vs. much lower zero-shot mAP on MSMT17). Same 512-D
   vector shape, drop-in at `detector._get_clip`/`_embed_tracks`. (arXiv 2503/CLIP-ReID; FastReID
   arXiv 2006.02631; DeepStream ReID.)

2. **Migrate `detection_events.bbox`+`meta` from jsonb to typed columns.** jsonb is TOASTed
   per-value and gets almost none of the columnar delta/XOR/Gorilla encoding that produces
   TimescaleDB's 90–95% compression; typed `float4`/`bool`/`int` columns do. This is both the
   largest storage win _and_ a scan-speed win for every analytics query.
   (tigerdata.com/blog/postgres-toast-vs-timescale-compression; compression-methods docs.)

3. **Persist per-track appearance in a dedicated `track_appearance` table (plan M4) instead of
   stashing 960+512 floats in `detection_events.meta`.** This severs the last dependency of
   `/rederive` on raw rows, letting us drop raw _immediately per video_ instead of holding it 2
   days, and it is where the gated per-track embedding samples for mean-pairwise-cosine live.
   Mirrors NVIDIA Metropolis expiring `mdx-raw-*` while persisting derived analytics + a Milvus
   embedding store. (docs.nvidia.com/mms Analytics_Data.)

4. **Consolidate zone detection: replace YOLO-World + SAM 2 with a single YOLOE-26-seg model
   (text-prompt), drop SAM 2 entirely.** YOLOE beats YOLO-World-v2 by +10–11 AP on LVIS zero-shot,
   runs ~1.4× faster, emits instance masks in the same forward pass, and eliminates the fragile
   git-only `clip` auto-pip dependency that currently disables our open-vocab path offline. Our
   `mask_to_polygon` already collapses masks to ≤12 points, so SAM 2's pixel precision is discarded
   anyway. (docs.ultralytics.com/models/yoloe; arXiv 2503.07465, 2602.00168.)

5. **Add quality-gated embedding + intra-tracklet DBSCAN split upstream of `build_raw_tracks`.**
   Our median pooling has no quality gate, so one occluded/truncated crop drags the identity
   vector off; BoT-SORT/StrongSORT gate ingestion by detector confidence. Splitting a
   `raw_track_id` on per-box embeddings (GTA's Tracklet Splitter) attacks the teacher id-steal at
   its source, upstream of the `teacher_chain.py` post-hoc patching. (GTA arXiv 2411.08216.)

**Explicitly NOT doing:** no Druid/Pinot (they target user-facing sub-second p99 at high
concurrency we don't have); no ClickHouse yet (only pays off under multi-month raw retention at
fleet scale); no FAISS/HNSW/Milvus (our gallery is tens of identities in one room — brute-force
cosine is exact and faster than maintaining an index).

---

## 2. Data storage

### 2.1 Verdict on the engine

**Stay on TimescaleDB as the system of record.** At the fleet target (80 cams × 8h × 5fps × ~30
objects ≈ **346M detection rows/day**, ~700M steady-state under 2-day retention) TimescaleDB
handles the volume with full SQL, transactional overlay writes, the idempotency/fence machinery
already in `db.replace_detections`, and continuous aggregates. ClickHouse's ~20× density and ~3×
ingest only pay off if we decide to **retain raw detections for months** at fleet scale
(multi-billion rows) — at which point it belongs as a _cold analytical mirror_, not a replacement.
Druid/Pinot are over-engineered here. Our existing three-tier instinct is correct; the work below
_hardens_ it rather than replacing it. (sanj.dev clickhouse-timescaledb comparison; StarTree "tale
of three OLAP dbs"; imply.io druid-vs-clickhouse.)

### 2.2 The core problem with the current schema

`detection_events` (`apps/api-service/src/db/schema/index.ts:91`) stores:

```
bbox  jsonb  -- {x, y, w, h}
meta  jsonb  -- {standing, back_to_camera, raw_track_id, hist?[960], embed?[512]}
```

Compression is configured (`drizzle/0003_storage_tiering.sql`):

```sql
timescaledb.compress_segmentby = 'video_id',
timescaledb.compress_orderby   = 'video_ts_ms, track_no'
```

But **jsonb defeats it**. TimescaleDB's columnar codecs (delta / delta-of-delta / simple-8b / RLE
for ints & timestamps, XOR/Gorilla for floats) only fire on typed columns; jsonb goes through
per-value TOAST + a dictionary that only helps if whole values repeat, so jsonb rows are ~5–20×
larger and barely compress. The reported 90–95% (10–20×) numbers are contingent on typed columns.
(tigerdata.com/blog/postgres-toast-vs-timescale-compression; compression-methods docs — _high
confidence_.)

### 2.3 Target schema (typed columns)

```sql
-- Target detection_events. bbox/meta jsonb -> typed columns.
CREATE TABLE detection_events (
  ts             timestamptz NOT NULL DEFAULT now(),  -- wall-clock partition key
  school_id      int         NOT NULL,                -- NEW: first-class tenant
  video_id       uuid        NOT NULL,
  video_ts_ms    bigint      NOT NULL,                -- per-video-relative time
  track_no       int         NOT NULL,                -- post-merge identity
  raw_track_id   int         NOT NULL,                -- was meta.raw_track_id
  bx             real        NOT NULL,                -- bbox.x  (was jsonb)
  by             real        NOT NULL,                -- bbox.y
  bw             real        NOT NULL,                -- bbox.w
  bh             real        NOT NULL,                -- bbox.h
  confidence     real        NOT NULL,
  standing       boolean     NOT NULL DEFAULT false,  -- was meta.standing
  back_to_camera boolean     NOT NULL DEFAULT false,  -- was meta.back_to_camera
  meta           jsonb                                -- residual: rare/sparse only
);
```

**bbox → four `real` columns** is the headline change. Per-detection appearance (`hist`, `embed`)
leaves `meta` entirely (§2.5). `meta` survives only as a residual for genuinely sparse/rare flags,
so the hot path no longer TOASTs a payload on every row.

Estimated effect: typed + compressed raw goes from heavy jsonb to roughly **single-digit
bytes/row**, i.e. ~2 GB/day compressed, ~4 GB at 2-day retention — and analytics scans read far
fewer bytes. (_High impact, medium effort._)

### 2.4 Compression, partitioning, multi-tenant

- **Keep the hypertable keyed on wall-clock `ts`, 1h chunks.** This is already correct
  (`0003_storage_tiering.sql`) — per-video `video_ts_ms` chunks never age, so
  compression/retention/cagg policies could never fire. Do not regress this.
- **`segmentby`:** for our VOD pattern each video is written once and never appended, so
  `segmentby = video_id` gives large, well-filled segments (~540k rows for a 1h lesson ≫ the
  ≥1,000-rows/segment floor). Distinct `video_id`s per wall-clock hour stays inside the ideal
  100–10,000 range. **Keep `video_id` as `segmentby`;** add `school_id` to `segmentby` only if it
  stays <10k distinct/chunk (it will). (compression guidance — _high confidence_.)
- **`orderby = (video_ts_ms, track_no)`** is already right: high-cardinality `track_no` belongs in
  the orderby prefix (run-length/delta gains), never in `segmentby`.
- **Multi-tenant isolation must be structural.** Add `school_id` as a first-class column on
  `detection_events`, `tracks`, `events`, `video_analytics`, `zones`, `track_appearance`, and the
  `track_minute` cagg. Enforce with row-level security or per-school schemas; a leading `school_id`
  also enables prefix pruning and per-school retention/tiering. Cheap now, expensive to retrofit
  after billions of rows. Once schools grow, consider space-partitioning by `school_id` hash.
  (ClickHouse `PARTITION BY (tenant, date)`; Parquet lake path `tenant/camera/date`; ES per-day
  indices — _high confidence_.)

### 2.5 Rollups vs. raw retention — decouple appearance from raw

Today `/rederive` still needs raw rows because the per-track median `hist` (~960 floats) and
`embed` (512 floats) are stashed in the **first raw row's `meta`** (`db.replace_detections:89–108`,
read back by `fetch_track_hists`/`fetch_track_embeds` via `meta ? 'hist'`). The
`0006_fast_hot_tier.sql` comment already flags the fix: _"persist the ~42-row per-track appearance
summary separately (see docs plan M4)."_ Do it:

```sql
CREATE TABLE track_appearance (
  school_id     int   NOT NULL,
  video_id      uuid  NOT NULL,
  raw_track_id  int   NOT NULL,
  n_dets        int   NOT NULL,
  hist          real[],        -- 960-float median torso HSV histogram
  embed         real[],        -- 512-float L2-normalized median re-ID vector
  embed_samples real[],        -- NEW: a few gated samples, flattened, for
  n_samples     int,           --      mean-pairwise-cosine in merge (§3.5)
  PRIMARY KEY (video_id, raw_track_id)
);
```

This is a plain permanent Postgres table (not a hypertable). Consequences:

- **`/rederive` no longer touches raw rows** — it reads `track_appearance` + the overlay tier. So
  raw retention can drop from 2 days to **immediate per-video** (or a short reprocessing buffer),
  reclaiming the hot tier faster.
- Materialize _all_ durable analytics before raw expires. We already do: `track_minute` cagg
  (`0003`), `video_analytics`, `tracks`/`events`, and the overlay polyline/keyframes in
  `tracks.meta`. Verify the cagg's ~1h refresh lag stays below raw retention (it does at 2 days;
  guard it if we go sub-day). This is the documented continuous-aggregate + independent-retention
  pattern, mirroring Metropolis expiring `mdx-raw-*` while persisting `mdx-behavior-*`/`mdx-mtmc-*`.
  (_High confidence._)

**Formalize the three tiers as a written lifecycle (SLA per tier):**

| Tier                 | Contents                                                                                                                   | Storage               | Lifetime                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------- | ----------------------------------------------------- |
| **Hot**              | raw `detection_events`, uncompressed rowstore                                                                              | Postgres rowstore     | last ~1h (live scrub / re-merge window)               |
| **Warm**             | raw `detection_events`, compressed columnar 1h chunks                                                                      | Hypercore columnstore | until raw retention (≤2 days, target: per-video drop) |
| **Cold / permanent** | overlay polyline+keyframes (`tracks.meta`), `track_appearance`, `video_analytics`, `tracks`, `events`, `track_minute` cagg | typed Postgres tables | forever                                               |

Naming the SLA prevents accidental reliance on raw and matches every reference system's
hot/warm/cold split (Hypercore tiered storage; Druid/Pinot deep storage; ClickHouse TTL tiers).

### 2.6 Optional cold object-storage tier (defer)

Only if a real replay/audit requirement for raw beyond retention appears: export aged compressed
chunks to Parquet on object storage partitioned
`s3://…/school_id=<>/video_id=<>/date=YYYY-MM-DD/`, queryable in place via DuckDB/Trino. Managed
transparent S3 tiering is Tiger Cloud-only; self-hosted needs this export job. Gives bottomless raw
retention at ~$0.02/GB-mo with partition-prefix deletes. Otherwise dropping raw is cheaper still —
**do not build speculatively.** (tigerdata data-tiering docs; Dremio Parquet-in-data-lakes — _high
effort, medium impact_.)

### 2.7 Migration path (storage)

1. **Add `school_id`** (backfill from a `videos.school_id` FK; default a sentinel tenant for
   existing rows). New Drizzle migration.
2. **Rebuild `detection_events` with typed columns.** TimescaleDB can't alter a hypertable's
   partitioning dimension in place — same rename→create→copy→drop pattern already used in
   `0003_storage_tiering.sql`. New table has `bx/by/bw/bh` + `standing`/`back_to_camera`/
   `raw_track_id`/`school_id` typed; re-apply `compress_segmentby='video_id'`,
   `compress_orderby='video_ts_ms, track_no'`, the 1h compression policy, and the 2-day retention
   policy. Re-create the `track_minute` cagg (add `school_id` to its GROUP BY) and `occupancy_minute`.
3. **Create `track_appearance`;** move the `hist`/`embed` writes out of `db.replace_detections`
   into an upsert on this table; repoint `fetch_track_hists`/`fetch_track_embeds` to read it.
4. **Update the ML COPY path** (`db.COPY_COLUMNS`, `replace_detections` record tuple): emit
   `bx/by/bw/bh/standing/back_to_camera/raw_track_id/school_id` as scalars instead of
   `json.dumps(bbox)` / `json.dumps(meta)`. Update `fetch_detections` to read scalar columns back
   into the `Detection` dataclass (`models.py:22`) — its shape is unchanged.
5. **Drop raw retention** toward per-video once (3) lands and `/rederive` is verified raw-free.
6. Keep micro-batch COPY (already `COPY_BATCH_SIZE = 5_000`, well under TimescaleDB's ceiling);
   consider Direct-Compress for reanalysis backfills. (_Low effort, medium impact._)

---

## 3. Person re-ID architecture

Production re-ID stacks decouple three things our pipeline currently fuses: a **re-ID-trained
backbone**, a **quality-gated per-identity feature bank**, and an **association layer** whose logic
changes with scale and appearance ambiguity. DeepStream is the canonical reference: a TensorRT
ResNet-50 ReIdentificationNet emitting 256-D L2-normalized embeddings, extracted _periodically_
(not per-detection), matched nearest-neighbor-in-gallery by cosine. Our greedy heap merge is
structurally offline **Global Tracklet Association (GTA)**; our seat-anchor veto + spatial-continuity
Gaussian are the textbook-correct instinct for the **degenerate identical-uniform regime**, where
sports/multi-camera SOTA likewise down-weight appearance and let motion/geometry dominate.

### 3.1 Backbone: CLIP ViT-B/32 → re-ID-fine-tuned + TensorRT

**Drop-in point:** `detector.py` — `CLIP_MODEL_NAME = "ViT-B/32"` (line 53), `_get_clip()`
(line 348), `_embed_tracks()` (line 390).

- **Preferred:** OSNet-AIN x1.0 (512-D, tiny, ONNX/TensorRT-ready) or CLIP-ReID ViT-B/16 (512-D —
  _same vector shape we already store_, so `track_appearance.embed` and `merge` math are unchanged).
- Export to ONNX→TensorRT FP16 (matches how `config.py` already exports YOLO26 to `.engine`); keep
  embeddings L2-normalized so dot product == cosine (already relied on in `_score_clusters:398`).
- Keep `build_raw_tracks` median pooling and re-norm (`merge.py:162–172`) **unchanged** — only the
  source of the per-crop vector changes.
- **Recalibrate `EMBED_VETO_COS` (0.35)** from the _new_ model's intra-vs-inter cosine histogram;
  a re-ID backbone widens the margin so the veto and the appearance term do real work instead of
  the HSV histogram and heuristics. Threshold should be per-deployment, not a hardcoded constant.
  (FastReID; CLIP-ReID; DeepStream ReID — _high confidence_.)

### 3.2 Quality-gated embedding ingestion

**Drop-in point:** `detector._extract_frame` (~lines 503–540), where crops/hists are appended per
detection. Before a crop feeds `crops`/`hists`, gate it: drop detections below a confidence floor,
with the bbox touching a frame edge (truncated), or with area/aspect outside person norms; then
median-pool only survivors. Optionally weight the median by confidence.

Our median currently has _no_ quality gate, so one occluded/truncated crop shifts both the identity
vector and the torso histogram, degrading every downstream cosine. BoT-SORT/StrongSORT set their
EMA update weight from detector confidence exactly to reject degraded crops; the AICity-2024
winner's "state-aware Re-ID correction" trusts appearance only in reliable track states. Cheapest
reliability win in the stack. (_Low effort, high impact._)

### 3.3 Intra-tracklet split before merge (GTA Tracklet Splitter)

**Drop-in point:** a new pass between `detect_video` output and `build_raw_tracks` (`merge.py:136`).

Cluster a `raw_track_id`'s per-detection embeddings (DBSCAN, cosine `eps ≈ 0.4–0.6`,
`min_samples ≈ 5`, ≤3 clusters, outliers reassigned to nearest); if it splits into
well-separated temporal segments, emit them as separate raw tracks (fresh `raw_track_id`s). This
requires retaining per-box embeddings for the split window — feed them from the gated samples of
§3.2 before they are pooled.

BoT-SORT id-steals routinely contaminate one `raw_track_id` with two people — the exact chimera
`teacher_chain.py` patches _after the fact_. Splitting upstream removes the chimera at the source,
lets the merge stage reassemble cleanly, and likely lets us retire some teacher-stitching
special-casing. This is the component that produces GTA's largest accuracy gain in the identical-
jersey (== identical-uniform) setting. (GTA arXiv 2411.08216 — _high impact, medium effort_.)

### 3.4 Gallery + ANN indexing — explicitly none

Keep **brute-force cosine** over the in-memory tracklet set. Our gallery is tens of identities in
one classroom per VOD; an ANN index (FAISS HNSW / IVF-PQ / Milvus) adds accuracy loss and code
surface for zero latency benefit at this scale. Metropolis reaches for Milvus only at the
1000+-camera / 3400-person MTMC regime. Add a one-line note at the top of `merge.py` documenting the
crossover point (~thousands of identities or multi-camera fusion) where an ANN index becomes
justified, to prevent premature optimization. (Metropolis MTMC; FAISS billion-scale docs — _high
confidence_.)

### 3.5 Online-windowed vs. global association

**Keep the whole-VOD global merge** — for offline VOD it is the right regime and is exactly GTA's
offline design (build a tracklet-pair distance matrix, hard-forbid temporally-overlapping merges,
gate by spatial displacement, greedy/hierarchical merge to a threshold). Our `merge_tracks` lazy-
deletion heap (`merge.py:494`) with the `_overlap_ms` overlap veto is the same family. Two refinements
to steal:

1. **Mean-pairwise-cosine, not median-vs-median.** In `_score_clusters` (`merge.py:396–408`) `cos`
   is a single median-vs-median dot product, brittle to pose. Store a few gated samples per raw
   track (`track_appearance.embed_samples`, §2.5) and compute cluster distance as the **mean
   pairwise cosine** over member samples — GTA's more-robust metric, what makes its hierarchical
   merge reliable under uniform appearance. Modest storage (a handful of 512-D vectors/track).
2. **State-aware appearance weight.** Scale the appearance term down when both tracks' mean
   confidence/visibility is low (AICity "state-aware Re-ID correction"), rather than a fixed weight.

(_Medium effort, medium impact._)

### 3.6 Degenerate identical-uniform regime — keep geometry-first, demote HSV

Our geometry-first design is correct and matches how sports/multi-camera SOTA handle identical
uniforms: motion + location dominate, appearance is a correction. **Keep** the hard seat-anchor
veto (`SEATED_VETO_DIST`, `merge.py:383–388`) and the spatial-continuity Gaussian
(`spatial_continuity`, line 211). Two changes:

- **Demote the HSV torso histogram from a co-equal appearance term to a tie-breaker** once the
  re-ID embedding is in place. Today appearance = `0.5*cos + 0.5*hist_correlation` (`merge.py:408`)
  with `W_HIST = 0.35`. A discriminative re-ID embedding subsumes torso color; keeping HSV at 0.5
  weight lets _identical_ uniform color inflate appearance and chain wrong fragments — the exact
  chimera the module docstring warns about. Reduce the hist weight, fold freed weight into
  spatial/temporal.
- Make appearance influence conditional on observation quality (§3.5.2), not fixed.

(_Low effort, medium impact._)

### 3.7 db.py / merge.py drop-in summary

| Change                          | File · anchor                                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| New backbone + TensorRT         | `detector.py:53 CLIP_MODEL_NAME`, `_get_clip:348`, `_embed_tracks:390`                                       |
| Quality-gate crops              | `detector._extract_frame` ~503–540                                                                           |
| Intra-tracklet DBSCAN split     | new pass before `merge.build_raw_tracks:136`                                                                 |
| Mean-pairwise cosine            | `merge._score_clusters:396–408`; needs `embed_samples`                                                       |
| Demote HSV / state-aware weight | `merge.py:52 W_HIST`, `_score_clusters:404–414`                                                              |
| Recalibrate veto                | `merge.py:85 EMBED_VETO_COS`                                                                                 |
| Per-track embeddings table      | `track_appearance` (§2.5); `db.replace_detections:89–108`, `fetch_track_hists:184`, `fetch_track_embeds:206` |
| No ANN index (documented)       | one-line note atop `merge.py`                                                                                |

---

## 4. Zone detection

**Verdict: migrate to a single YOLOE-26-seg model in text-prompt mode; drop SAM 2; keep the
geometric score as the trust gate; keep fine-tuning in reserve.** YOLOE-26-seg is confirmed present
in our installed Ultralytics.

### 4.1 Why

- **Accuracy + speed:** YOLOE beats YOLO-World-v2 by +3.5 AP on LVIS with ~1.4× faster inference
  and ~⅓ the training cost; YOLOE26-L beats YOLO-World-L by +10.0 AP and runs ~6.2 ms / 161 FPS on
  a T4 — squarely our deploy class. (docs.ultralytics.com/models/yoloe; arXiv 2503.07465,
  2602.00168 — _high confidence_.)
- **One model, masks free:** YOLOE emits instance masks in the same forward pass (`no additional
overhead`), collapsing our two-model YOLO-World→SAM 2 chain. Class-labeled masks also move
  board-vs-door disambiguation _into the model_ instead of leaning entirely on `score_mask`/
  `score_door` geometry.
- **SAM 2 is wasted here:** `mask_to_polygon` (`board_detect.py:207`) simplifies every mask to
  4–12 points via `approxPolyDP`, discarding SAM 2's pixel precision. Coarse static zones only need
  "tight enough" masks, which YOLOE provides. Dropping SAM 2 removes the ~20-point grid inference
  per frame (`GRID_*`/`DOOR_GRID_*`), a second model's memory, and the MPS/CPU-fallback complexity
  (`_sam_segment:310`).
- **Kills the offline-fragility bug:** text-prompt YOLO-World needs the git-only `clip` package,
  whose absence currently disables our entire open-vocab path (`_get_world:274`, the
  `importlib.util.find_spec("clip")` guard) and silently degrades to `sam2_geometric`. Use YOLOE
  text-prompt as primary, and the **prompt-free `yoloe-26*-seg-pf`** variant (no CLIP/MobileCLIP
  text encoder) as the offline-safe fallback in place of the `_world_failed` guard.

**Text prompt over prompt-free for primary:** prompt-free is bounded to a fixed ~4,585-class
LVIS+Objects365 vocabulary and snaps unusual objects to the nearest label; our `BOARD_PROMPTS`
("chalkboard", "projection screen") / `DOOR_PROMPTS` ("doorway") are specialized enough to need
attribute-precise text queries. Use `-pf` only on the offline fallback path.
(learnopencv YOLOE tutorial; arXiv 2602.00168 — _high confidence_.)

**Fine-tuning: reserve, don't lead.** A small fine-tuned YOLO26-seg would win on accuracy (published
gains up to tens of AP on domain data) but costs ~100–300 labeled frames + training (~8h vs ~10min
zero-shot). For a run-once, human-corrected static zone the open-vocab + geometric + HITL stack is
almost certainly sufficient. Trigger fine-tuning only after logging YOLOE open-vocab miss-rate on
real classroom footage; if triggered: `freeze=10`, ~50 epochs, patience 15, reduced mosaic.
(ultralytics finetuning-guide; Tenyks zero-shot-vs-finetune — _medium confidence_.)

### 4.2 Migration (board_detect.py)

1. **Replace the model layer.** Swap `_get_world`/`_get_sam` for a single lazy `_get_yoloe()` that
   loads `yoloe-26l-seg` and calls `set_classes(WORLD_PROMPTS)` (reuse the existing
   `BOARD_PROMPTS`+`DOOR_PROMPTS`, `N_BOARD_CLASSES` split unchanged). Export to TensorRT on the GPU.
2. **Rewrite `_detect_on_frame`/`_detect_door_on_frame`:** run YOLOE once per frame; take polygons
   directly from `result.masks` per class group; feed each through the **unchanged**
   `score_mask`/`score_door` + `mask_to_polygon` path. The whole `MIN_SCORE`/`DOOR_MIN_SCORE` +
   geometric-gate contract and the `DetectBoardResponse` shape (`models.py:223`) stay intact.
3. **Delete SAM 2 surface:** `SAM_MODEL_NAMES`, `_get_sam`, `_sam_segment`, `_dedupe_masks`,
   `GRID_*`/`DOOR_GRID_*`, and the SAM CPU-fallback state.
4. **Prompt-free fallback:** if the text-encoder weights are missing, load `yoloe-26l-seg-pf`
   instead of failing — replaces the `clip`-absent `_world_failed` degradation.
5. **Temporal voting (cheap upgrade):** we already sample 3 frames (`FRAME_FRACTIONS`) but take a
   raw `max` (`detect_board:500`). Upgrade to IoU-clustered voting: cluster candidate polygons
   across the sampled frames, pick the cluster with the most agreement + highest median score, and
   **persist the agreement count as a stability signal.** Static-camera literature shows multi-frame
   voting suppresses shake/ghost false positives. (mdpi 12/15/3346 — _medium confidence_.)
6. **HITL confidence (schema tie-in):** store the zone confidence + a low-confidence flag (below
   `MIN_SCORE`/`DOOR_MIN_SCORE`, or single-frame-only) into `zones.meta`
   (`ZoneMeta { auto?, confidence?, method? }` already exists, `schema/index.ts:5`) and surface it
   in the video-detail UI for one-time human accept/adjust. Zones are set once per video, so a
   cheap human confirm eliminates silent mis-detections propagating into all downstream board-time/
   circulation analytics. (HITL ROI best practice — _high confidence_.)

---

## 5. Ranked roadmap

Ship-order is dependency- and leverage-ordered. Prior art in the last column.

| #   | Item                                                                                      | Area    | Effort | Impact | Prior art                                                                       |
| --- | ----------------------------------------------------------------------------------------- | ------- | ------ | ------ | ------------------------------------------------------------------------------- |
| 1   | Quality-gate crops before median pooling (`_extract_frame`)                               | re-ID   | low    | high   | BoT-SORT/StrongSORT confidence-gated EMA; AICity-2024 state-aware ReID          |
| 2   | `detection_events` bbox+meta jsonb → typed columns                                        | storage | medium | high   | Hypercore compression contingent on typed cols (tigerdata TOAST-vs-compression) |
| 3   | Add `school_id` first-class + RLS across all derived tables                               | storage | medium | high   | ClickHouse `PARTITION BY (tenant,date)`; Parquet `tenant/…` path                |
| 4   | `track_appearance` table; move hist/embed out of raw `meta`; enable per-video raw drop    | storage | medium | high   | Metropolis expires `mdx-raw-*`, persists derived + Milvus embeds                |
| 5   | Replace CLIP ViT-B/32 with OSNet-AIN / CLIP-ReID + TensorRT; recalibrate `EMBED_VETO_COS` | re-ID   | medium | high   | FastReID BNNeck; CLIP-ReID; DeepStream ReIdentificationNet                      |
| 6   | YOLOE-26-seg text-prompt replaces YOLO-World+SAM 2; drop SAM 2; `-pf` offline fallback    | zones   | medium | high   | YOLOE CVPR-2025 (arXiv 2503.07465 / 2602.00168); docs.ultralytics               |
| 7   | Intra-tracklet DBSCAN split before `build_raw_tracks`                                     | re-ID   | medium | high   | GTA Tracklet Splitter (arXiv 2411.08216)                                        |
| 8   | Mean-pairwise cosine + state-aware appearance weight in `_score_clusters`                 | re-ID   | medium | medium | GTA distance metric; AICity-2024                                                |
| 9   | Demote HSV torso hist to tie-breaker (`W_HIST`)                                           | re-ID   | low    | medium | GTA/sports: appearance uninformative under identical uniforms                   |
| 10  | Formalize hot/warm/cold lifecycle SLA; verify cagg lag < retention                        | storage | low    | medium | Hypercore tiered storage; Druid/Pinot deep storage                              |
| 11  | IoU-clustered temporal voting for zones + persist agreement                               | zones   | low    | medium | static-region temporal-voting literature (mdpi)                                 |
| 12  | Zone confidence → `zones.meta`; HITL confirm in UI                                        | zones   | medium | medium | HITL ROI frameworks                                                             |
| 13  | Micro-batch / Direct-Compress ingest tuning for backfills                                 | storage | low    | medium | ClickHouse 3–4M rows/s; Timescale Direct-Compress                               |
| 14  | Document "no ANN index" crossover note in `merge.py`                                      | re-ID   | low    | low    | Metropolis Milvus only at 1000+-cam scale                                       |
| 15  | (Reserve) Parquet object-store cold tier for raw audit/replay                             | storage | high   | medium | Tiger Cloud S3 tiering; Dremio Parquet lakes                                    |
| 16  | (Reserve) Fine-tune YOLO26-seg on labeled board/door                                      | zones   | high   | medium | Ultralytics finetuning-guide; Tenyks few-shot                                   |

**Not on the roadmap (decided against):** Druid / Pinot (over-engineered for internal dashboards);
ClickHouse as system-of-record (revisit only as a cold mirror under multi-month raw retention at
fleet scale); FAISS / HNSW / Milvus (brute-force cosine is exact and faster at tens of identities).

---

### Sources (representative)

TimescaleDB Hypercore & tiering: tigerdata.com/docs/build/columnar-storage/setup-hypercore ·
tigerdata.com/blog/postgres-toast-vs-timescale-compression · docs.tigerdata.com data-tiering.
NVIDIA: docs.nvidia.com/mms Analytics_Data / MDX_Multi_Camera_Tracking · DeepStream ReID
(ridgerun/developer.nvidia.com multi-camera-tracking). Re-ID: GTA arXiv 2411.08216 · CLIP-ReID /
TransReID / SOLIDER · FastReID arXiv 2006.02631 · AICity-2024 Track-1 (CVPRW). Zones: YOLOE arXiv
2503.07465 & 2602.00168 · docs.ultralytics.com/models/yoloe · learnopencv YOLOE tutorial.
OLAP comparisons: sanj.dev · StarTree · imply.io. Full URL list retained in the research appendix.
