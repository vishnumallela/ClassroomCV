# Luminary — Optimal storage & database architecture

**Status:** decision · **Owner:** lead data/infra architect · **Scope:** `detection_events` hypertable + overlay/rollup tiers, MinIO/S3 video, the live 80-camera ingest path
**Grounding:** `docs/architecture-decision.md` §5, `docs/storage-and-reid-design.md` §2, `apps/api-service/src/db/schema/index.ts`, `drizzle/0003_storage_tiering.sql`, `drizzle/0006_fast_hot_tier.sql`

**Bottom line up front.** The repo's instinct is already correct and the market agrees: **store the event and the tracklet, not the frame.** The single decision that keeps storage bounded is not the database engine — it is retention: per-frame `detection_events` is a *disposable hot buffer* whose durable outputs are (a) compressed tracklets, (b) discrete events, and (c) continuous-aggregate rollups. Do that and your permanent footprint drops ~100× and stays flat regardless of camera count. Two honest amendments the adversarial reviews force on the naive version of this plan: (1) keep a **bounded raw window** (days, not "drop after aggregation") so you can recompute metrics, audit false negatives, and retrain; (2) keep **continuous video for the legal minimum**, not just event clips. Stay on TimescaleDB now; pre-wire a ClickHouse cold mirror off-ramp but do not build it yet.

---

## What the market does

Every deployed large-scale video-analytics system converges on the same shape: **record full video cheaply at the edge, ship only structured metadata + thumbnails to the central store, fetch pixels on demand.** Nobody centralizes every frame's detections long-term — the thing Luminary's per-frame `detection_events` tier does by default is the one thing the entire market deliberately avoids persisting.

The specific proof points:

- **Verkada** records full continuous video to on-camera storage (30–365-day SKUs), and in steady state streams only **encrypted thumbnails + metadata every 5–20 s at 20–50 Kbps per camera** — full video is fetched on demand when a user scrubs the timeline. The cloud is a thumbnail + metadata index, not a frame store. (verkada.com/blog/scale-your-camera-deployment; help.verkada.com storage FAQ)
- **Eagle Eye Networks** buffers the heavy raw stream on an on-site Bridge and uploads policy-tiered per camera/time/resolution; edge analytics "generate descriptions (metadata)" that the cloud indexes — not frames. (een.com)
- **Ambient.ai** persists only validated "signals"/events after an explicit relevance filter — a **90–95% reduction** in what is surfaced/stored downstream. Discard benign activity at ingest. (ambient.ai/blog)
- **NVIDIA DeepStream/Metropolis** (the canonical 80+-camera reference, mirrored in your own docs) runs detection + single-camera tracking at the edge, emits a thin event-centric metadata schema (`{timestamp, sensorId, event, object, bbox, single-cam track id, unified track id, state}`) over **Kafka**; **Cassandra** holds trajectory/event state keyed by track, **Elasticsearch** is the search index, **video never enters the metadata path**, and embeddings are generated only at a configurable `skip-interval`, never per frame. Reference deployment: 150 fisheye + 8 LPR cameras at 2 FPS, 0.5 s cross-camera polling. (developer.nvidia.com/blog/multi-camera-large-scale-iva-deepstream-sdk; docs.nvidia.com/vss object-detection-tracking)

The concrete numbers that make this non-negotiable: continuous full video is **~7–15 GB per camera per day** at 1080p/H.265 — for your 80-cam × 8h target that's ~0.5–0.6 TB/day of *video alone*. A standard detection metadata record (timestamp, camera, class, bbox, confidence, zone, event, duration) is **so compact it can be retained months–years after the underlying video is overwritten**, each row linking to a clip for verification. (safetyscope.eu on-prem guide; safetyscope.eu/glossary/video-metadata-surveillance) "The shift is from pixels to events and embeddings." That is exactly the three-tier plan already sketched in your docs — the market validates it; the leverage is in retention, not the engine.

---

## What to store (and what to drop)

The ranking by bytes, small → large: `discrete events ≪ tracklet polylines < rollups ≪ raw per-frame detections ≪ embeddings ≪ thumbnails ≪ event clips ≪ full video`. The design decision is to make the **permanent record** the cheap left side and let the expensive right side expire on a clock.

Per-camera-hour sizing (1 cam-hour = 3600 s × 5 fps × ~30 people = **540k detections**, matching your stated 540k rows/1h; annualized figures are for the full 80-cam × 8h/day fleet):

| Data class | Keep? | Representation | Retention window | Tier | ~Size / cam-hour |
|---|---|---|---|---|---|
| **Raw per-frame detections** | **Bounded, then drop** | Typed columns (`int2` bbox `x,y,w,h`, `int2` scaled conf, `bool` standing/back, `int` track ids) — NOT jsonb; Hypercore columnstore, segmentby `video_id` | **Hot 1h rowstore → warm columnar to 72h–30d**, then drop (jurisdiction-set; see below) | Timescale hypertable | ~150 MB jsonb → **~5–10 MB typed+compressed** |
| **Tracklets / trajectories** | **Keep (permanent)** | Per-track RDP/TD-TR polyline + 2 s keyframes + standing/back **state-change** points (not per-frame) in `tracks.meta` | **Forever** | Typed Postgres (non-hyper overlay) | **<100 KB** |
| **Events** (enter/exit/board/teacher-transition) | **Keep (permanent)** | One typed row per event + confidence + `raw_track_id` linkage | **Forever** | Postgres `events` | **<2 KB** |
| **Rollups** (occupancy/dwell/instructional-time) | **Keep (permanent)** | `track_minute` cagg (HLL for unique-people) + per-lesson `video_analytics` | **Forever** | Timescale cagg + typed table | **~50 KB** |
| **Embeddings** (CLIP/seat re-ID) | **Ephemeral** | One median 512-D vector **per tracklet**, in-session only; seat-anchor geometry is the durable re-ID, not a stored template | **In-video / evict post-lesson** | RAM / `track_appearance`, then delete | ~60 KB (never persisted long-term) |
| **Thumbnails** | **Keep (index)** | 1 keyframe / 5–20 s, small JPEG — the timeline never needs the hot tier to render | **= video window** | MinIO/S3 | ~5 MB |
| **Event clips** | **Keep (verification)** | Short H.265 clips around discrete events (DeepStream "Smart Record") | **= or > video window** | MinIO/S3 | ~45 MB (≈5% of footage) |
| **Full continuous video** | **Legal-minimum only** | H.265 1080p, presigned-URL fetch; lifecycle → cold archive | **30–90 days** (edge/cold), **not clips-only** | Edge NVR → S3/Glacier | **~900 MB** |

**The explicit decisions:**

1. **Collapse frames → tracklets → events as the permanent record.** Persisting one polyline per track instead of 540k rows/cam-hour is a **~5–20× cut** (RDP alone gives ~80% point reduction, TD-TR variants preserve the timestamps you need for dwell/mobility, and a streaming variant — Opening-Window/SQUISH-E/BQS — does it online for live). Combined with dropping raw, the durable footprint falls **~100×** versus keeping per-frame rows forever. (trajectory-compression survey, onlinelibrary.wiley.com/10.1155/2016/6587309; online BQS arXiv 1412.0321; DeepStream nvdsanalytics)

2. **"Drop raw after aggregation" — but with a bounded window, not immediately.** The adversarial review is correct that literal drop-after-rollup is refutable on four grounds: (a) aggregation is lossy and irreversible, so any *new* metric you invent next quarter (proximity, turn-taking) would need ML re-run on video you may also have dropped; (b) tracklets are a lossy summary and your own re-ID is error-prone (your recent commits fight teacher id-steals) — fixing a mislabel later needs the per-frame bboxes/embeddings; (c) MLOps drift-retraining consumes raw features, not rollups; (d) event-triggered clips structurally **cannot audit false negatives** (a missed event writes no clip → you can measure precision, never recall). So: keep raw for a **rolling window aligned to your accuracy-review + retrain cadence** (14–30 days is the defensible engineering choice), then compress hard and keep only tracklets + rollups. (evidentlyai.com data-drift; TrADe Re-ID arXiv 2209.06452; statisticsbyjim.com aggregation weaknesses)

3. **The retention window is a jurisdictional knob, and privacy pulls it *shorter* while retraining/legal pull it *longer* — resolve it explicitly.** GDPR/EDPB reasoning says raw CCTV "should be erased after a few days," and beyond **72 h** you must document why (edpb.europa.eu Guidelines 3/2019). US school practice and FERPA preservation-holds push continuous video to **30–90 days** (Texas: 3 months minimum), because disputes surface days later and don't map to a detector event (ecam.com; coram.ai school-surveillance). These are not contradictory if you split the two buckets: **raw detection *rows*** get the short privacy clock (72 h–30 d); **continuous *video*** gets the legal minimum at the edge/cold store with a preservation-hold override. Make both per-deployment policy, and **never persist a biometric template** — a stored faceprint imports BIPA's 3-year destruction duty + $1k–5k *per student* exposure that lands on the school (forasoft.com BIPA; Rosenbach). Seat-anchor geometry re-ID, ephemeral embeddings: keep this a headline product guarantee.

4. **Store pose/skeleton/event metadata, not appearance.** This is the rare case where the privacy-minimal representation is *also* the cost-minimal one — skeleton sequences are reported at **<1% of source-image storage** (arxiv 2411.14565). Add **k-anonymity suppression** on small-headcount rooms (e.g. <5 students in a zone) so a "1 student at the board" stat can't re-identify.

---

## Which database & tech

**Verdict: stay on TimescaleDB as the system of record. Do not adopt Druid/Pinot ever for this workload. Pre-wire ClickHouse as a hybrid-CDC *cold mirror* off-ramp, and switch only at a specific, measurable threshold — which you are nowhere near today.**

This is not vibes; the two adversarial verifications quantify it.

**Ingest is not your forcing function — it cannot justify a migration.** Your 346M rows/day is **~12k rows/s sustained**, which is ~3% of proven single-node TimescaleDB capacity. TimescaleDB 2.21 Direct-to-Columnstore sustains >5M records/s; the vendor's reference service reached **350 TB / 1 trillion+ rows ingesting 10B/day — ~29× your daily volume** (tigerdata.com 350TB case; TimescaleDB 2.21 release). ClickHouse ingests faster (~2–4M rows/s) but **punishes small batches** (250 ops/s at batch-100 vs Timescale's 14,200) so it *requires* a Kafka buffer — a second system's worth of ops for a problem you don't have. (tinybird.co/blog/clickhouse-vs-timescaledb)

**Your dominant query patterns favor TimescaleDB, not ClickHouse.** Luminary reads are per-lesson rollups (`video_analytics`), overlay tracklets, occupancy caggs, and re-ID joins across `detections↔tracks↔zones↔lessons` — a *normalized, join-heavy, point-lookup* workload. On RTABench (relational time-series with joins, closer to your schema than ClickBench's flat scans) **TimescaleDB is 1.9× faster** than ClickHouse, and on last-point/most-recent-row queries it wins **0.6s vs 4.6s** — ClickHouse's structural weak spot. Plus PG-native operational simplicity is a real advantage for a small team; ClickHouse self-hosting demands distributed-systems expertise. (tinybird.co; Cloudflare ran analytics reporting on TimescaleDB precisely to avoid a second store, blog.cloudflare.com/timescaledb-art)

**Druid/Pinot are unambiguously over-engineered here.** They target user-facing sub-100ms p99 at high external concurrency; a minimal Druid cluster is 4–5 coordinated process types. Your consumers are internal dashboards. If you ever leave Postgres, the correct target is **ClickHouse (single binary, smallest ops surface)**, not Druid. (oneuptime.com clickhouse-vs-druid)

**The exact flip threshold — migrate a ClickHouse cold mirror when ANY of these breaks** (all hold today):
- **Retained *queryable* raw rows cross ~1B.** At 80 cams you hit 1B raw rows in **~3 days** — so this trips the moment you decide to keep raw per-frame detections queryable past ~3 days *and* run scans across them. The bounded-window design (raw dropped/compressed after your 14–30d review window, aggregates serving history) keeps the *live-queryable* raw set well under this. (dev.to observability benchmark: B-tree index misses lose to columnar scans at billion-row scale)
- **Analytics go ad-hoc.** Continuous aggregates pre-materialize your known metrics; ClickHouse becomes correct when you need *interactive ad-hoc aggregation across full raw history* that caggs can't pre-compute — concretely, when raw-hypertable aggregations routinely exceed a few seconds, or PG connections saturate during reporting peaks. (oneuptime.com migrate-from-timescaledb)
- **High-cardinality dims enter the hot path.** Cross-camera re-ID queries over millions of distinct `raw_track_id`/embedding keys — TimescaleDB's 557k→159k rows/s cardinality cliff. Mitigated today because per-classroom person cardinality is low and bounded.

When it flips, the market pattern is **not rip-and-replace** — it's **hybrid CDC**: keep writes + recent + join-heavy data in Timescale, replicate the historical detection subset into ClickHouse via Postgres CDC (ClickPipes), offloading reads first and the write pipeline last, for 20–40× compression + fast wide group-bys. (clickhouse.com/blog/timescale-to-clickhouse-clickpipe-cdc)

**The non-database pieces (adopt now):**
- **Kafka (or Redpanda) backbone, keyed by `camera_id`** — one camera → one partition → one stateful consumer (never split a camera mid-stream; tracking is stateful). This is the single change that takes you from 1 video to 80 live cameras without rewiring, and it moves tracklet/re-ID state into checkpointed Flink/Kafka-Streams consumers instead of the DB — matching your existing perception-vs-analytics split. **At-least-once + idempotent dedup-on-write** (`cameraId+frame_ts+track_id`) for high-rate detections; reserve **transactional exactly-once** for the low-rate enter/exit/board `events`. (kafka.apache.org; confluent.io)
- **MinIO/S3 for video + thumbnails + clips**, presigned-URL fetch, kept out of the metadata path (your MinIO tier is correct). Self-hosted MinIO only beats tiered S3 on $/TB past *hundreds of TB with ops staff* — not at today's scale.
- **No vector DB / ANN index yet.** Your gallery is tens of identities in one room; brute-force cosine is exact and faster. Metropolis reaches for Milvus only at 1000+-camera MTMC. Document the crossover, don't build it.

---

## Cost & scale math

**Sizing formulas (give these to the team):**
```
rows/day    = cameras × hours × 3600 × fps × avg_persons_per_frame
DB bytes/day = rows/day × raw_bytes_per_row ÷ compression_ratio
blob TB/day  = cameras × hours × 3600 × bitrate_Mbps ÷ 8 ÷ 10^6
```

**Two buckets, sized separately.**

**(A) Blob tier — raw video (dominates raw TB, bounded by lifecycle policy not DB design):**
2 Mbps × 3600 s = 900 MB/cam-hour → 8h = 7.2 GB/cam/day → ×80 = **~576 GB/day ≈ 0.58 TB/day ≈ 17 TB/month ≈ 210 TB/year**.
Cost of a 1-year rolling raw corpus (210 TB), us-east-1:
- All S3 Standard ($0.023/GB): **~$4,830/mo**
- Lifecycle: 7-day hot Standard (~4 TB, ~$95) + rest Deep Archive (~206 TB × $0.00099 ≈ $208): **~$300/mo** — a **~16× cut from tiering alone** (Standard→Deep-Archive is 23×, "the highest-ROI cost decision").
- Event clips (~5%) + thumbnails only: **<$50/mo**.
Set an S3 lifecycle rule; don't hand-roll it. (cloudburn.io/blog/amazon-s3-pricing; aws.amazon.com/s3/pricing)

**(B) Row tier — detections (dominates query cost; where jsonb blows up or typed stays bounded):**
80 cams × 8h × 3600 × 5 fps × ~30 people ≈ **346M rows/day (~12k rows/s)**. Raw jsonb row ≈ 300 B → **~104 GB/day = ~38 TB/yr uncompressed**.
- **Postgres jsonb (1.5–3×):** ~19 TB/yr — jsonb repeats every key name on every row; expensive on any hot SSD.
- **TimescaleDB typed + Hypercore (10–15×):** ~2.5–3.2 TB/yr.
- **ClickHouse typed + DoubleDelta(ts)/Gorilla(coords)/LowCardinality(enums) (~15–20×):** ~2.3 TB/yr.

The jsonb→typed-columnar delta is **~8×**, entirely from dropping repeated jsonb keys and delta/dictionary-encoding columns you already have — the codecs map 1:1 onto your schema (monotonic `video_ts_ms`→DoubleDelta cuts timestamps >95%; float bbox→Gorilla ~12×; `standing`/`back_to_camera`→LowCardinality dictionary 30×+). **This is the single change that keeps the row tier bounded, and you get it inside TimescaleDB without ClickHouse.** (clickhouse.com database-compression; tigerdata TOAST-vs-compression)

**The retention multiplier that makes it flat, not growing:**
The above is per *year of retained raw*. With the bounded-window design, live-queryable raw never exceeds ~30 days:
- Raw hot+warm at **30-day** retention (Timescale 12×): 346M × 30 × ~25 B ≈ **~260 GB steady-state**, not 3 TB/yr.
- Tracklets + events + rollups (permanent): grow at **<100 KB × 640 cam-hours/day ≈ ~64 MB/day ≈ ~23 GB/yr** — trivially retained forever.
- Video at 30–90-day legal window with lifecycle → cold: **~50–100 TB rolling, ~$300–600/mo**.

**All-in target: ~$400–700/month** (video lifecycle + a single-node Timescale instance + MinIO), and it stays flat as you add years because the permanent tier is the ~23 GB/yr metadata, not the 38 TB/yr raw.

---

## Gap vs current design + migration path

**What the repo already gets right** (do not regress):
- Hypertable keyed on **wall-clock `ts`, 1h chunks**, compress after 1h, `compress_segmentby='video_id'`, `compress_orderby='video_ts_ms, track_no'` (`0003_storage_tiering.sql`) — textbook-correct. Per-video `video_ts_ms` chunks would never age; keeping `ts` is what lets policies fire.
- The **three-tier instinct**: raw hot → overlay polyline+keyframes in `tracks.meta` → `video_analytics` + `track_minute` cagg. This is the market pattern.
- The `run_tokens`/`StaleRunError` fence and micro-batch COPY (`COPY_BATCH_SIZE=5000`) — the idempotency machinery you'll extend for windowed live ingest.
- `docs/storage-and-reid-design.md` already reaches the right verdicts: stay TimescaleDB, no Druid/Pinot, ClickHouse only as a cold mirror, no ANN index, typed columns, `track_appearance` table.

**What diverges from optimal:**
- **`detection_events.bbox` and `.meta` are jsonb** (`schema/index.ts:95,97`). jsonb is TOASTed per-value and gets almost none of the columnar delta/XOR/Gorilla encoding — so the configured 90–95% compression **barely fires**, and every hot dashboard read parses JSON per row. This is the ~8× storage miss and the top-priority fix.
- **`/rederive` and the golden-eval path still depend on raw rows**, because the per-track median `hist` (960 floats) and `embed` (512 floats) are stashed in the *first raw row's `meta`* (read via `fetch_track_hists`/`fetch_track_embeds`). This couples raw retention to re-derivation — you can't drop raw fast while this holds. `0006_fast_hot_tier.sql` already flags the fix.
- **No explicit tracklet-compression tier as the durable overlay** beyond keyframes in `tracks.meta` — formalize the RDP/TD-TR polyline as the survives-raw-drop visual record.
- **Retention is 2 days** (per storage doc) — fine for VOD; the live fleet needs the explicit hot(1h rowstore)/warm(columnar to 30d)/cold(forever metadata) SLA plus the video legal-window policy, written down (GDPR requires you to *demonstrate* retention justification).
- **No Kafka ingest / no windowed append writer** — `replace_detections` (whole-video DELETE+COPY) cannot serve an unbounded 8h live stream (`architecture-decision.md` §4 Stage 2 already scopes this).

**Ordered, low-risk migration path** (⚠️ = touches the golden-eval / re-derivation path — gate behind a golden-eval regression run):

1. **Retype `detection_events` jsonb → typed columns** (`int2` bbox `bx,by,bw,bh`, `int2` scaled conf 0–1000, `bool` standing/back_to_camera, `int` raw_track_id; residual `meta` jsonb only for rare flags). TimescaleDB can't alter a partitioning dimension in place — use the rename→create→copy→drop pattern already in `0003`. Re-apply compression + retention policies, re-create `track_minute`/`occupancy_minute` caggs. ~8× storage + scan-speed win. **⚠️** updates the COPY path (`db.COPY_COLUMNS`, `replace_detections`) and `fetch_detections` read-back into the `Detection` dataclass — shape unchanged, but exercise the golden eval.
2. **Create `track_appearance` table; move `hist`/`embed` out of raw `meta`.** **⚠️** repoint `fetch_track_hists`/`fetch_track_embeds`; this is what severs `/rederive` from raw rows. **This is the load-bearing step for everything after** — do it before touching retention. Verify the golden eval reproduces identical re-derived tracks reading from `track_appearance` + overlay instead of raw.
3. **Formalize the tracklet overlay as the durable tier** — RDP/TD-TR polyline + state-change points in `tracks.meta` (or a plain non-hyper `track_overlay` table), guaranteed to survive raw retention. Materialize *all* durable analytics before raw expires; verify cagg refresh lag < raw retention.
4. **Bounded drop-raw-after-window.** Once (2)+(3) land and `/rederive` is verified raw-free, set raw retention to the accuracy-review/retrain window (14–30 d, jurisdiction-set), *longer* than the cagg refresh lookback so aggregates never refresh over deleted ranges. Add the S3 lifecycle rule for video (7-day hot → Deep Archive) + preservation-hold hook. **Do not** go clips-only on video — keep continuous video for the legal minimum.
5. **Object-store tiering + thumbnails-as-index** — ensure the UI timeline renders from thumbnails, never the hot detection tier; event clips via Smart-Record on discrete events only.
6. **Kafka ingest, keyed by `camera_id`** + the append-only idempotent windowed writer (`architecture-decision.md` §4 Stages 1–2). At-least-once detections with offset-in-DB dedup; exactly-once for enter/exit/board events. Keep `replace_detections` for VOD unchanged.
7. **(Defer, do not build yet) ClickHouse cold mirror via Postgres CDC** — wire the interface but trip it only at the measured threshold: retained-queryable raw > ~1B rows, or raw aggregations routinely > a few seconds, or PG connections saturating. Reads first, writes last.

**Sequencing note:** steps 1–2 are the highest-leverage and are golden-eval-sensitive; land them first with regression coverage. Steps 3–5 bound storage. Step 6 unblocks the 80-camera live fleet. Step 7 is a future off-ramp, not present work.