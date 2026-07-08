import { createFileRoute } from "@tanstack/react-router";
import type { CSSProperties } from "react";
import { Card } from "@/components/ui/card";

export const Route = createFileRoute("/architecture")({ component: Architecture });

// --------------------------------------------------------------------------- //
// Technical reference for the ML pipeline. Values are the real constants from
// services/ml-service (detector.py, merge.py, roles.py, events.py, quality.py)
// and apps/api-service/drizzle. Terse by intent: this is a spec, not a story.
// --------------------------------------------------------------------------- //

const SUMMARY: [string, string][] = [
  ["Input", "classroom video (H.264/H.265, any length), + optional board/door zones"],
  [
    "Output",
    "teacher presence / board / entry-exit intervals, 5 s occupancy series, 32x18 dwell heatmap, per-lesson analytics, data-quality report",
  ],
  [
    "Sampling",
    "~5 fps (stride = round(native_fps / 5)); full frame rate is redundant for behaviour",
  ],
  ["Compute", "NVIDIA GPU (CUDA fp16 / TensorRT); MPS for dev; one durable worker per video"],
  ["Privacy", "no facial recognition; skeleton + ephemeral re-ID only; aggregate persistence"],
];

// Ordered data-flow. Each stage consumes the previous stage's output.
const PIPELINE: { n: string; stage: string; module: string; io: string }[] = [
  {
    n: "0",
    stage: "Decode + sample",
    module: "detector.iter_frames",
    io: "video file -> (ts_ms, BGR frame) at ~5 fps",
  },
  {
    n: "1",
    stage: "Detect + pose",
    module: "YOLO26-pose",
    io: "frame -> N x {bbox, conf, 17 keypoints}",
  },
  { n: "2", stage: "Track", module: "BoT-SORT", io: "per-frame boxes -> boxes + raw_track_id" },
  {
    n: "3",
    stage: "Per-detection features",
    module: "detector._extract_frame",
    io: "box + keypoints -> standing, back_to_camera, torso HSV hist, CLIP crop",
  },
  {
    n: "4",
    stage: "Embed",
    module: "detector._embed_tracks (CLIP ViT-B/32)",
    io: "crops -> 512-d L2-normalised median embedding per raw track",
  },
  {
    n: "5",
    stage: "Merge (re-ID)",
    module: "merge.merge_tracks",
    io: "raw tracks + hists + embeds -> raw_id -> identity map (1..N)",
  },
  {
    n: "6",
    stage: "Teacher-chain stitch",
    module: "teacher_chain.stitch_teacher",
    io: "reclaim teacher spans stolen by student ids; evict bad merges",
  },
  {
    n: "7",
    stage: "Role assignment",
    module: "roles.assign_roles",
    io: "identity features -> teacher | student | unknown + confidence",
  },
  {
    n: "8",
    stage: "Event derivation",
    module: "events.derive",
    io: "roles + zones -> presence / board / entry-exit / occupancy / heatmap",
  },
  {
    n: "9",
    stage: "Data quality",
    module: "quality.assess",
    io: "detections + roles -> coverage, fragmentation, concurrent count, tiers",
  },
  {
    n: "10",
    stage: "Persist",
    module: "db.replace_detections + API",
    io: "TimescaleDB hypertable (tiered) + typed analytics tables",
  },
];

const MODELS = [
  {
    name: "YOLO26-pose",
    family: "Detection + 2D pose, single-stage",
    task: "Per frame, locate every person and estimate their 17-keypoint skeleton in one forward pass.",
    mechanism:
      "YOLO family: whole-image single pass, no region proposals. YOLO26 is NMS-free (one-to-one head, no duplicate-suppression step) and uses RLE keypoint regression for tighter joint localisation. +up to 7.2 pose AP over YOLO11.",
    solves:
      "The only stage that reads raw pixels. The 17 keypoints are load-bearing: hip-above-knee geometry and box aspect drive the standing/seated signal, so teacher vs student is decided with no face and no name.",
    input: "1 BGR frame, 2560x1440",
    output: "<=100 detections x { bbox[x,y,w,h] norm, conf, keypoints[17][x,y,conf] }",
    params: "imgsz 1280-1536 · fp16 · conf>=0.1 · max_det 100 · yolo26x (GPU) / yolo26m (dev)",
  },
  {
    name: "BoT-SORT",
    family: "Multi-object tracking (tracking-by-detection)",
    task: "Assign a temporally stable id to each person by linking boxes across frames. Never reads pixels.",
    mechanism:
      "Kalman filter predicts each track's next state from motion; detections are matched to predictions by IoU via the Hungarian algorithm; a ByteTrack second pass associates low-confidence boxes. GMC (camera-motion comp.) disabled: static camera.",
    solves:
      "Same-person-over-time. Every temporal metric (presence, movement, entries, board time) needs a continuous trajectory, not disconnected boxes.",
    input: "per-frame boxes",
    output: "boxes + raw_track_id (trajectories)",
    params:
      "gmc none · track_buffer 60 (~12 s) · new_track_thresh 0.40 · match_thresh 0.8 · fuse_score false",
  },
  {
    name: "CLIP ViT-B/32 + HSV hist + seat anchors",
    family: "Re-identification (appearance + geometry scoring)",
    task: "Merge track fragments (from occlusion / exit) back into stable identities.",
    mechanism:
      "Per candidate pair: score = 0.35 appearance + 0.25 spatial + 0.20 size + 0.20 temporal. appearance = 0.5 cos(CLIP) + 0.5 HSV-hist corr. Hard vetoes: cos < 0.35 => different; two seated fragments (centre range < 0.02) with anchors > 0.10 apart => different. Greedy merge above 0.55.",
    solves:
      "Fragmentation. Under identical uniforms appearance is near-degenerate, so the discriminator is location: a seat is the one cue a uniform cannot fake. CLIP's high-value roles are the veto and re-linking the distinctly-dressed teacher. Embeddings are ephemeral (single lesson).",
    input: "raw tracks + <=10 upper-body crops each + torso hists",
    output: "raw_track_id -> identity (1..N by first appearance)",
    params: "MERGE_THRESHOLD 0.55 · EMBED_VETO_COS 0.35 · MAX_GAP 10 min · overlap tol 1 s",
  },
  {
    name: "YOLOE-26-seg",
    family: "Open-vocabulary detection + segmentation, one model",
    task: "Auto-detect the board and door polygons (run once, up front).",
    mechanism:
      "An open-vocabulary model in the YOLO26 family that DETECTS and SEGMENTS in one pass: a text prompt ('chalkboard', 'classroom door') returns the region and a mask together, replacing the older YOLO-World + SAM 2 two-model chain. A geometric score (aspect, wall height, rectangularity, colour) picks the winner; a SAM 2 grid-probe fallback remains for when the text encoder is unavailable.",
    solves:
      "Turns 'standing at the front' into board time and 'vanished near a door' into a confirmed exit. Operator can redraw either zone.",
    input: "representative frame + text prompts",
    output: "board / door polygons (normalised points)",
    params: "yoloe-26s-seg · score threshold 0.25 · +10-11 AP and ~1.4x faster than YOLO-World",
  },
];

// Per-detection features (detector.py). Everything downstream is derived from these.
const FEATURES: [string, string][] = [
  [
    "standing",
    "bbox aspect h/w > 1.6, OR hip-above-knee keypoints (kpt conf > 0.4, box h >= 90/1440); smoothed by a 5-sample majority vote before use",
  ],
  ["back_to_camera", "face keypoints (nose/eyes) conf < 0.3 while both shoulders conf > 0.5"],
  [
    "torso HSV histogram",
    "cv2.calcHist over (H,S), 30x32 bins, ranges [0,180]x[0,256], L1-normalised; region from torso keypoints or bbox 0.2-0.8 w x 0.15-0.6 h",
  ],
  ["CLIP crop", "upper 60% of bbox, downscaled to <=224 px, <=10 samples/track >=1 s apart"],
];

// Teacher/student signals (roles.py). Weighted, then an outlier decision rule.
const SIGNALS: { label: string; weight: string; def: string; teacher: string; student: string }[] =
  [
    {
      label: "standing_ratio",
      weight: "0.30",
      def: "fraction of samples standing (see feature above)",
      teacher: "0.74",
      student: "0.05",
    },
    {
      label: "movement",
      weight: "0.25",
      def: "max(x_range, y_range) of bbox-centre trajectory / 0.40, clamped to 1",
      teacher: "0.98",
      student: "0.04",
    },
    {
      label: "presence_ratio",
      weight: "0.25",
      def: "(last_ms - first_ms) / duration_ms",
      teacher: "0.98",
      student: "0.90",
    },
    {
      label: "board_proximity",
      weight: "0.20",
      def: "fraction standing, centre in board x-range, box bottom below board floor; dropped + re-weighted if no board zone",
      teacher: "0.19",
      student: "0.00",
    },
  ];

const GATES: { t: string; rule: string; d: string }[] = [
  {
    t: "min span",
    rule: "span >= 60 s",
    d: "short fragments / passers-by cannot be teacher (scaled down for clips < 3 min)",
  },
  {
    t: "not frame-edge",
    rule: "0.03 < mean cx,cy < 0.97",
    d: "clipped edge boxes always read as tall/standing",
  },
  {
    t: "not tiny",
    rule: "mean_area >= 0.3 x median",
    d: "a distant back-row head cannot outscore the adult",
  },
];

const EVENTS: [string, string][] = [
  [
    "presence",
    "union of teacher detection timestamps; split at gaps >= 5 s; off-camera gaps <= 12 s away from any door are bridged",
  ],
  [
    "entries / exits",
    "presence-interval edges where any sample within a 2 s window is inside a door zone (expand 0.15); video-start counts as enter, final interval into last 5 s produces no exit",
  ],
  [
    "board",
    "hysteresis state machine: 2 s sustained ON to open, 3 s OFF to close; a >= 5 s sampling gap hard-closes; tolerates single-frame flicker (budget 600 ms)",
  ],
  ["occupancy", "distinct non-teacher track_no per 5 s bucket -> avg_students, max_students"],
  ["heatmap", "32x18 grid of bbox-centre dwell counts, teacher vs student"],
];

const QUALITY: [string, string][] = [
  [
    "coverage",
    "occupied 5 s buckets / span buckets over [first, last] detection; tiers >=0.9 / >=0.7",
  ],
  ["fragmentation", "raw_tracks / identities; tiers <=2 (clean) / <=4 (fair) / else low"],
  [
    "concurrent count",
    "p95 of non-teacher boxes per frame; re-id-INDEPENDENT cross-check on max_students (one body = one box per frame)",
  ],
  [
    "confidence tiers",
    "per dimension (coverage, tracking, occupancy, teacher) + overall = weakest link",
  ],
];

const STORAGE: { tier: string; contents: string; policy: string }[] = [
  {
    tier: "Hot",
    contents: "raw per-frame detection_events (TimescaleDB hypertable, wall-clock ts, ~1 h chunks)",
    policy:
      "compress after 1 h (static post-write), drop after 2 days; only needed for cheap /rederive",
  },
  {
    tier: "Overlay",
    contents:
      "per-track RDP centre polyline (eps 0.005) + bbox keyframes every 2 s, in tracks.meta",
    policy: "permanent; ~2% of raw; serves playback after hot rows age out",
  },
  {
    tier: "Aggregate",
    contents: "events, track summaries, video_analytics, 1-min occupancy continuous aggregate",
    policy: "permanent; negligible size; everything the dashboard reads",
  },
  {
    tier: "Media",
    contents: "uploaded video + thumbnail bytes (blobs, not DB rows)",
    policy:
      "local disk (dev) or S3-compatible object storage; on-prem MinIO keeps student video on-site, cloud S3/R2 by config; worker caches a local copy for ffmpeg/ML",
  },
];

const BOUNDARIES: [string, string][] = [
  [
    "Time-on-task via interval sampling; in-seat vs out-of-seat; presence / head-count",
    "Student engagement / attention / focus / motivation as a validated construct or mental state (restricted in EU education under AI Act Art. 5(1)(f))",
  ],
  [
    "Gross movement, teacher circulation, dwell spread, image-plane coverage",
    "Emotion / affect / mood from face or body (no validated mapping; Barrett et al. 2019)",
  ],
  [
    "Standing vs seated posture with detector confidence attached",
    "Head/body orientation as gaze or attention (orientation-toward-board proxy only)",
  ],
  [
    "Detection/tracking accuracy (mAP, HOTA, quality tiers)",
    "Demographic fairness for per-individual scoring (no subgroup validation)",
  ],
  [
    "Aggregate, zone-level, teacher-facing reflection; ephemeral within-session re-ID",
    "Per-student attendance register, longitudinal per-child profile, or persisted appearance biometric",
  ],
  [
    "Proximity / dwell in metres AFTER a one-time camera homography, with error bars",
    "Pixel proximity as comparable across cameras/rooms; distance in metres without calibration",
  ],
];

const STACK: [string, string][] = [
  ["Frontend", "Vite, TanStack Router + Query, shadcn, Tailwind"],
  ["API", "Bun, Hono, oRPC, Drizzle"],
  ["Queue", "BullMQ on Redis"],
  ["ML service", "FastAPI, Ultralytics YOLO26-pose, BoT-SORT, CLIP, YOLOE-26-seg, ffmpeg"],
  ["Store", "TimescaleDB (hypertable + continuous aggregates + compression/retention)"],
  ["Media", "Local disk or MinIO / S3 / R2 for video + thumbnail blobs (Bun native S3 client)"],
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="reveal space-y-4">
      <h2 className="font-display text-xl font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

// Two-column key/value spec table.
function KV({ rows, keyClass = "w-44" }: { rows: [string, string][]; keyClass?: string }) {
  return (
    <Card className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <tbody className="divide-y divide-border">
            {rows.map(([k, v]) => (
              <tr key={k} className="align-top">
                <td className={`${keyClass} px-4 py-2.5 font-mono text-[0.8rem] font-medium`}>
                  {k}
                </td>
                <td className="px-4 py-2.5 leading-relaxed text-muted-foreground">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Architecture() {
  return (
    <div className="mx-auto max-w-4xl space-y-12">
      <header className="reveal space-y-4">
        <div className="font-mono text-xs uppercase tracking-widest text-primary">
          ML pipeline · technical reference
        </div>
        <h1 className="font-display text-3xl font-semibold tracking-tight">
          How the analytics are computed
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Video in, structured teaching analytics out. A fixed camera is sampled at 5 fps and passed
          through detection, tracking, re-identification, role classification, and event derivation.
          No frame is labelled by hand and no face is recognised. Constants below are the ones the
          service actually runs.
        </p>
        <KV rows={SUMMARY} keyClass="w-28" />
      </header>

      {/* Pipeline */}
      <Section title="1 · Pipeline">
        <p className="text-sm leading-relaxed text-muted-foreground">
          Eleven stages, each consuming the previous output. Stages 1-4 run per frame in the ML
          service; 5-9 run once over the whole video; 10 writes the tiered store.
        </p>
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-border bg-muted/40 font-mono text-[0.7rem] uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Stage</th>
                  <th className="px-3 py-2 font-medium">Module</th>
                  <th className="px-3 py-2 font-medium">In &rarr; Out</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {PIPELINE.map((s) => (
                  <tr key={s.n} className="align-top">
                    <td className="px-3 py-2.5 font-mono text-primary">{s.n}</td>
                    <td className="px-3 py-2.5 font-medium">{s.stage}</td>
                    <td className="whitespace-nowrap px-3 py-2.5 font-mono text-[0.72rem] text-muted-foreground">
                      {s.module}
                    </td>
                    <td className="px-3 py-2.5 font-mono text-[0.72rem] leading-relaxed text-muted-foreground">
                      {s.io}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </Section>

      {/* Models */}
      <Section title="2 · Models">
        <div className="stagger space-y-4">
          {MODELS.map((m, i) => (
            <Card key={m.name} className="p-5" style={{ "--i": i } as CSSProperties}>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="font-mono text-base font-semibold">{m.name}</h3>
                <span className="rounded-md bg-muted px-2 py-0.5 font-mono text-[0.7rem] text-muted-foreground">
                  {m.family}
                </span>
              </div>
              <dl className="mt-3 space-y-2.5 text-sm leading-relaxed">
                <SpecRow k="Task" v={m.task} />
                <SpecRow k="Mechanism" v={m.mechanism} />
                <SpecRow k="Solves" v={m.solves} />
                <SpecRow k="Params" v={m.params} mono />
              </dl>
              <div className="mt-3 grid gap-2 border-t border-border pt-3 sm:grid-cols-2">
                <IoBox label="in" v={m.input} />
                <IoBox label="out" v={m.output} />
              </div>
            </Card>
          ))}
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">
          Two later stages are explainable rules, not neural models: the role classifier (section 4)
          and the event deriver (section 5). Keeping them as rules means every number traces to a
          cause.
        </p>
      </Section>

      {/* Features */}
      <Section title="3 · Per-detection features">
        <p className="text-sm leading-relaxed text-muted-foreground">
          Computed once per person per frame; every analytic is derived from these plus the track
          id.
        </p>
        <KV rows={FEATURES} />
      </Section>

      {/* Re-ID */}
      <Section title="4 · Re-identification merge">
        <KV
          rows={[
            ["candidate", "temporal overlap < 1 s AND gap < 10 min"],
            ["score", "0.35 appearance + 0.25 spatial + 0.20 size + 0.20 temporal"],
            [
              "appearance",
              "0.5 cos(CLIP) + 0.5 HSV-hist corr (both present), else spatial continuity",
            ],
            [
              "hard veto",
              "cos < 0.35 => different; seated pair (range < 0.02) with anchors > 0.10 => different",
            ],
            ["merge", "greedy max-heap, threshold 0.55; identities numbered by first appearance"],
            [
              "teacher rescue",
              "adult-height + mobile pair gets an appearance floor when embeds agree (leave-and-return)",
            ],
          ]}
          keyClass="w-36"
        />
      </Section>

      {/* Teacher classification */}
      <Section title="5 · Teacher classification (roles.py)">
        <p className="text-sm leading-relaxed text-muted-foreground">
          Each eligible identity gets a weighted composite over four behavioural signals; the
          teacher is the clear outlier or nobody. Representative teacher/student values from the
          demo lesson.
        </p>
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-border bg-muted/40 font-mono text-[0.7rem] uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">signal</th>
                  <th className="px-3 py-2 font-medium">w</th>
                  <th className="px-3 py-2 font-medium">definition</th>
                  <th className="px-3 py-2 text-right font-medium">teacher</th>
                  <th className="px-3 py-2 text-right font-medium">student</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {SIGNALS.map((s) => (
                  <tr key={s.label} className="align-top">
                    <td className="px-3 py-2.5 font-mono text-[0.78rem] font-medium">{s.label}</td>
                    <td className="px-3 py-2.5 font-mono text-primary">{s.weight}</td>
                    <td className="px-3 py-2.5 leading-relaxed text-muted-foreground">{s.def}</td>
                    <td className="px-3 py-2.5 text-right font-mono tabular-nums">{s.teacher}</td>
                    <td className="px-3 py-2.5 text-right font-mono tabular-nums text-muted-foreground">
                      {s.student}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div className="overflow-x-auto rounded-lg bg-muted/60 p-4 font-mono text-xs leading-relaxed">
          <div>score = 0.30 stand + 0.25 roam + 0.25 present + 0.20 board</div>
          <div className="mt-2 text-muted-foreground">
            # teacher only if the top score is a genuine outlier
          </div>
          <div>
            teacher = argmax if <span className="text-primary">best &gt;= 0.50</span> and{" "}
            <span className="text-primary">(best - 2nd) &gt;= max(0.08, 0.15 x best)</span>
          </div>
          <div className="text-muted-foreground">else: all unknown (degrade gracefully)</div>
          <div className="mt-2">role_confidence = min(1, 0.5 + margin)</div>
        </div>

        <p className="text-xs font-medium text-muted-foreground">
          Eligibility gates (before scoring):
        </p>
        <div className="grid gap-2 sm:grid-cols-3">
          {GATES.map((g) => (
            <div key={g.t} className="rounded-lg border border-border bg-background/50 p-3">
              <div className="text-sm font-medium">{g.t}</div>
              <code className="mt-1 block font-mono text-[0.72rem] text-primary">{g.rule}</code>
              <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{g.d}</p>
            </div>
          ))}
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">
          Post-selection, teacher_chain.stitch_teacher reclaims her fragments that student ids stole
          during walk-ins near the door / board / her own path, so she stays one identity across the
          lesson.
        </p>
      </Section>

      {/* Events */}
      <Section title="6 · Event derivation (events.py)">
        <KV rows={EVENTS} keyClass="w-32" />
      </Section>

      {/* Quality */}
      <Section title="7 · Data-quality report (quality.py)">
        <p className="text-sm leading-relaxed text-muted-foreground">
          Additive; never mutates a derived number. Quantifies how much to trust each figure.
        </p>
        <KV rows={QUALITY} keyClass="w-40" />
      </Section>

      {/* Storage */}
      <Section title="8 · Storage tiers">
        <p className="text-sm leading-relaxed text-muted-foreground">
          A 1-hour lesson at 5 fps is ~18k frames / ~540k detection rows. Three tiers keep the raw
          firehose bounded while the dashboard survives retention.
        </p>
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-border bg-muted/40 font-mono text-[0.7rem] uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">tier</th>
                  <th className="px-3 py-2 font-medium">contents</th>
                  <th className="px-3 py-2 font-medium">policy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {STORAGE.map((t) => (
                  <tr key={t.tier} className="align-top">
                    <td className="px-3 py-2.5 font-mono font-medium text-primary">{t.tier}</td>
                    <td className="px-3 py-2.5 leading-relaxed">{t.contents}</td>
                    <td className="px-3 py-2.5 leading-relaxed text-muted-foreground">
                      {t.policy}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </Section>

      {/* Boundaries */}
      <Section title="9 · Claim boundaries">
        <p className="text-sm leading-relaxed text-muted-foreground">
          Grounded in the affect-recognition evidence and EU AI Act / FERPA / GDPR. Left: measured
          and defensible. Right: invalid or restricted, therefore not computed or claimed.
        </p>
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-border bg-muted/40 text-[0.72rem] uppercase tracking-wider">
                <tr>
                  <th className="px-3 py-2 font-medium text-primary">CAN claim</th>
                  <th className="px-3 py-2 font-medium text-destructive/80">CANNOT claim</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {BOUNDARIES.map(([can, cannot]) => (
                  <tr key={can} className="align-top">
                    <td className="w-1/2 px-3 py-2.5 leading-relaxed">{can}</td>
                    <td className="w-1/2 px-3 py-2.5 leading-relaxed text-muted-foreground">
                      {cannot}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </Section>

      {/* Stack */}
      <Section title="10 · Stack">
        <KV rows={STACK} keyClass="w-28" />
      </Section>
    </div>
  );
}

function SpecRow({ k, v, mono = false }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[5.5rem_1fr] gap-3">
      <dt className="font-mono text-[0.72rem] font-medium uppercase tracking-wide text-muted-foreground">
        {k}
      </dt>
      <dd className={mono ? "font-mono text-[0.78rem] text-muted-foreground" : ""}>{v}</dd>
    </div>
  );
}

function IoBox({ label, v }: { label: string; v: string }) {
  return (
    <div className="rounded-lg bg-muted/50 px-3 py-2">
      <span className="font-mono text-[0.65rem] uppercase tracking-wider text-primary">
        {label}
      </span>
      <div className="mt-0.5 font-mono text-[0.72rem] leading-relaxed text-muted-foreground">
        {v}
      </div>
    </div>
  );
}
