import { createFileRoute } from "@tanstack/react-router";
import type { CSSProperties } from "react";
import { Card } from "@/components/ui/card";

export const Route = createFileRoute("/architecture")({ component: Architecture });

// --------------------------------------------------------------------------- //
// Technical reference for the ML pipeline. Values are the real constants from
// services/ml-service (detector.py, zones.py, teacher.py, heuristics.py,
// quality.py) and apps/api-service/drizzle. Terse by intent: this is a spec,
// not a story. The measurements behind each number are in docs/rfdetr-pipeline.md.
// --------------------------------------------------------------------------- //

const SUMMARY: [string, string][] = [
  ["Input", "classroom video (H.264/H.265, any length), + optional board/door zones"],
  [
    "Output",
    "the three teacher KPIs — entry/exit, board time, 32x18 dwell heatmap — plus presence/board intervals and a data-quality report",
  ],
  [
    "Sampling",
    "~5 fps (stride = round(native_fps / 5)); full frame rate is redundant for behaviour",
  ],
  ["Compute", "NVIDIA GPU (CUDA); MPS for dev; one durable worker per video"],
  [
    "Privacy",
    "no facial recognition, no appearance biometric, no student detected at all — the model looks for the teacher, the board and the door",
  ],
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
    stage: "Detect",
    module: "RF-DETR (fine-tuned, 5 classes)",
    io: "frame batch -> N x { class, bbox, conf }",
  },
  {
    n: "2",
    stage: "Zone gate",
    module: "zones.gate_static",
    io: "drop board/door boxes outside their zone; the teacher is never gated",
  },
  {
    n: "3",
    stage: "Follow the teacher",
    module: "teacher.build_teacher_track",
    io: "teacher boxes -> one timeline (continuity first, confidence second)",
  },
  {
    n: "4",
    stage: "Event derivation",
    module: "events.derive",
    io: "her timeline + zones -> presence / board / entry-exit / heatmap",
  },
  {
    n: "5",
    stage: "Data quality",
    module: "quality.assess",
    io: "her timeline -> coverage, timeline breaks, detection confidence, tiers",
  },
  {
    n: "6",
    stage: "Persist",
    module: "db.replace_detections + API",
    io: "TimescaleDB hypertable (tiered) + typed analytics tables — HER rows only",
  },
];

const MODELS = [
  {
    name: "RF-DETR (fine-tuned)",
    family: "Detection transformer, single-stage, NMS-free",
    task: "Per frame, locate the teacher, the board, the door, and board-interaction moments (pointing, writing).",
    mechanism:
      "A DETR-family transformer fine-tuned on this product's own five classes. Object queries attend over the whole frame and emit a set of boxes directly, with no region proposals and no duplicate-suppression step. Because 'teacher' is a trained class, the model NAMES her rather than leaving a downstream stage to infer which detected person is the adult.",
    solves:
      "The question every KPI hangs off. The previous pipeline detected people and then inferred the teacher from stature, limb proportions, uniform colour, CLIP appearance, a tracklet DP and a vision-model vote — about 2,750 lines of inference that a trained class replaces. Students are not a class, so no student is ever detected, stored or drawn.",
    input: "batch of BGR frames, 2560x1440",
    output: "N x { class_id, bbox[x,y,w,h] norm, conf }",
    params:
      "resolution 576 · teacher conf 0.4 · zone conf 0.5 · batch 16 (GPU) / 1 (dev)",
  },
  {
    name: "Continuity tracking",
    family: "Single-target association (no ByteTrack, no BoT-SORT, no Re-ID)",
    task: "Decide which box is hers when the model offers more than one, and whether she is still here across a frame where it offered none.",
    mechanism:
      "The chain seeds on the first instant carrying exactly ONE teacher box, then grows forwards and backwards. A candidate is admitted only if a person could physically have moved there: allowed distance = 0.05 + 0.8 x gap_seconds, measured from her own ground-truth speed. Past a 5 s gap the position gate switches off — she may have left and re-entered by another door.",
    solves:
      "Identity switching, without a tracker. At the configured threshold the model emits at most one teacher box per frame (0 frames with two, over 583 scored), and its longest unbroken miss is ~5.4 s. There is no second identity to confuse her with, so re-identification has nothing to re-identify.",
    input: "teacher-class boxes per sampled instant",
    output: "one timeline + a coverage/confidence reading",
    params: "MAX_SPEED_PER_S 0.8 · JUMP_BASE 0.05 · FREE_GAP_MS 5000",
  },
  {
    name: "Zone proposal",
    family: "Median over the lesson's own detections",
    task: "Place the board and door polygons for a room, then hold later detections to them.",
    mechanism:
      "The board and door do not move, and the detector finds them in almost every frame — measured presence 96-100% (board) and 77-81% (door), with a centre spread of +/-0.002 over an entire lesson. So each edge is the median of its own detections across the video, which ignores the occasional stray box a single frame would enshrine.",
    solves:
      "A room configures itself on first upload and the operator only corrects it. Gating later detections to the zone stops a wall poster that reads as a screen for six frames from moving the board.",
    input: "a sparse pass over the video (~0.1 fps)",
    output: "board / door polygons (normalised points) + a confidence",
    params: "MIN_PRESENCE 0.30 · MIN_SAMPLES 8 · GATE_TOLERANCE 0.05",
  },
];

// What the detector emits per box. Everything downstream is derived from these.
const FEATURES: [string, string][] = [
  ["class", "one of door · screen · teacher · pointing · writing"],
  ["bbox", "{x, y, w, h} normalised 0-1, top-left based, clamped to the frame"],
  ["conf", "the model's score; per-class thresholds are applied downstream so one pass serves every consumer"],
  [
    "track_no",
    "set only on the teacher's accepted detections — everything else carries none and is never stored",
  ],
];

// How the teacher timeline is decided (teacher.py).
const SIGNALS: { label: string; weight: string; def: string; teacher: string; student: string }[] =
  [
    {
      label: "class = teacher",
      weight: "required",
      def: "the model's own label; there is no student class to compete with",
      teacher: "0.88 (p50 conf)",
      student: "n/a",
    },
    {
      label: "continuity",
      weight: "first",
      def: "reachable from the previous accepted box at <= 0.05 + 0.8 x gap_s",
      teacher: "0.55 /s peak observed",
      student: "n/a",
    },
    {
      label: "confidence",
      weight: "second",
      def: "breaks ties only AMONG reachable candidates, never overrides continuity",
      teacher: "0.79 (p10)",
      student: "n/a",
    },
  ];

const GATES: { t: string; rule: string; d: string }[] = [
  {
    t: "teacher threshold",
    rule: "conf >= 0.40",
    d: "middle of a plateau: coverage is flat 0.15-0.60, and competing boxes vanish above 0.25",
  },
  {
    t: "plausible motion",
    rule: "dist <= 0.05 + 0.8 x gap_s",
    d: "a box that crosses the room between two samples is not her, however confident",
  },
  {
    t: "free after a gap",
    rule: "gap >= 5 s",
    d: "she may have left and re-entered elsewhere; position then carries no information",
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
  ["heatmap", "32x18 grid of the teacher's bbox-centre dwell counts"],
];

const QUALITY: [string, string][] = [
  [
    "teacher coverage",
    "sampled frames she was found in / frames sampled; tiers >=0.85 / >=0.6",
  ],
  [
    "continuity",
    "how many times her timeline broke; entries/exits are counted from those breaks, so dozens means that KPI is an upper bound; tiers <=6 / <=20",
  ],
  ["detection", "mean confidence of the detections behind her timeline; tiers >=0.7 / >=0.5"],
  ["confidence tiers", "per dimension (coverage, continuity, detection) + overall = weakest link"],
];

const STORAGE: { tier: string; contents: string; policy: string }[] = [
  {
    tier: "Hot",
    contents:
      "the teacher's per-frame detections (TimescaleDB hypertable, wall-clock ts, ~1 h chunks). No student, board or door rows exist",
    policy:
      "compress after 1 h (static post-write), drop after 2 days; only needed for cheap /rederive",
  },
  {
    tier: "Overlay",
    contents:
      "her RDP centre polyline (eps 0.005) + bbox keyframes every 2 s, in tracks.meta",
    policy: "permanent; ~2% of raw; serves playback after hot rows age out",
  },
  {
    tier: "Aggregate",
    contents: "events, track summary, video_analytics (three teacher KPIs), zone polygons",
    policy: "permanent; negligible size; everything the dashboard reads",
  },
  {
    tier: "Media",
    contents: "uploaded video + thumbnail bytes (blobs, not DB rows)",
    policy:
      "local disk (dev) or S3-compatible object storage (on-prem MinIO keeps student video on-site, cloud S3/R2 by config); probe/thumbnail read a presigned URL directly, the GPU worker fetches its own copy by allowlisted presigned URL, so nothing downloads the whole video onto the API node",
  },
];

const BOUNDARIES: [string, string][] = [
  [
    "Teacher circulation, dwell spread, image-plane coverage, time at the board",
    "Student engagement / attention / focus / motivation as a validated construct or mental state (restricted in EU education under AI Act Art. 5(1)(f))",
  ],
  [
    "Her entries and exits, counted at a configured door zone",
    "Emotion / affect / mood from face or body (no validated mapping; Barrett et al. 2019)",
  ],
  [
    "Board-interaction moments (pointing, writing) as detected events",
    "Head/body orientation as gaze or attention (orientation-toward-board proxy only)",
  ],
  [
    "Detection accuracy against per-frame ground truth (coverage, purity, quality tiers)",
    "Demographic fairness for per-individual scoring (no subgroup validation)",
  ],
  [
    "Aggregate, zone-level, teacher-facing reflection",
    "Anything about a student: none is detected, stored, or drawn, so there is no per-student register or profile to build",
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
  ["ML service", "FastAPI, RF-DETR (fine-tuned), PyTorch, ffmpeg"],
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
