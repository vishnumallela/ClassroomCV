/**
 * The end-to-end data flow as one SVG: ingestion, the video pipeline on the
 * GPU, the audio pipeline beside it, where both land, the read-time
 * derivations, and the UI. Drawn from a small node/edge spec so the picture
 * and the labels stay in one place; colours come from the theme tokens so it
 * reads in both themes.
 */

interface Node {
  id: string;
  x: number;
  y: number;
  w?: number;
  lines: string[];
  tone?: "default" | "store" | "ui" | "model";
}

interface Edge {
  from: string;
  to: string;
  label?: string;
  /** Route: straight right, or drop down then across. */
  via?: "right" | "down";
}

const W = 1480;
const H = 900;
const NODE_H = 54;
const NODE_W = 168;

const LANES: { y: number; h: number; title: string }[] = [
  { y: 16, h: 96, title: "1 · Ingestion — API (Bun) · object store" },
  { y: 128, h: 186, title: "2–5 · Video pipeline — ML service on the GPU pod (one job per video)" },
  { y: 330, h: 96, title: "6 · Audio pipeline — API worker · AssemblyAI (parallel, no GPU)" },
  { y: 442, h: 96, title: "Storage — Postgres + TimescaleDB · MinIO" },
  {
    y: 554,
    h: 176,
    title: "7–10 · Read time — API derives every number from stored rows (no re-run)",
  },
  { y: 746, h: 138, title: "12 · UI — React dashboard" },
];

const NODES: Node[] = [
  // lane 1
  { id: "cam", x: 24, y: 48, lines: ["Camera file", "MP4 · 25 fps · 45 min", "burned-in clock"] },
  {
    id: "upload",
    x: 224,
    y: 48,
    lines: ["Upload", "POST /videos (stream ≤ 4 GB)", "videos row · zone template"],
  },
  {
    id: "store",
    x: 424,
    y: 48,
    lines: ["Object store", "MinIO/S3 + local copy", "presigned URLs"],
    tone: "store",
  },
  {
    id: "probe",
    x: 624,
    y: 48,
    lines: [
      "Probe + thumbnail",
      "ffprobe: duration, fps, size,",
      "creation_time → recording start",
    ],
  },
  {
    id: "queues",
    x: 824,
    y: 48,
    lines: ["Queues (BullMQ / Redis)", "video: concurrency 1", "audio: concurrency 4"],
  },
  {
    id: "pod",
    x: 1064,
    y: 48,
    w: 190,
    lines: ["GPU pod (RunPod)", "created from Settings · /health", "must say cuda before a run"],
  },
  // lane 2 row 1
  {
    id: "fetch",
    x: 24,
    y: 152,
    lines: ["Fetch media", "presigned URL via tunnel", "allowlist · resume"],
  },
  {
    id: "decode",
    x: 224,
    y: 152,
    lines: ["Decode + sample", "cv2 every frame,", "keep 5 fps · ts_ms"],
  },
  {
    id: "detect",
    x: 424,
    y: 152,
    lines: ["RF-DETR (5 classes)", "batch 16 @ 576 fp16", "floor 0.15"],
    tone: "model",
  },
  {
    id: "desc",
    x: 624,
    y: 152,
    lines: ["Appearance", "HSV-16 of the torso band", "per teacher box ≥ 0.3"],
  },
  {
    id: "zones",
    x: 824,
    y: 152,
    lines: ["Zones", "propose board/door (median)", "gate static classes"],
  },
  {
    id: "track",
    x: 1024,
    y: 152,
    lines: ["Tracker → segments", "containment dedup · predicted", "position · reach 0.05+0.8/s"],
  },
  {
    id: "attr",
    x: 1224,
    y: 152,
    w: 230,
    lines: ["Attribution → people", "swaps · split 0.42 · link 0.30", "handed-over rule → track 1"],
    tone: "model",
  },
  // lane 2 row 2
  {
    id: "events",
    x: 1224,
    y: 244,
    w: 230,
    lines: ["Events + analytics", "presence · door IN/OUT · board", "pointing/writing · heatmap"],
  },
  {
    id: "person",
    x: 1024,
    y: 244,
    lines: ["Per-person analytics", "presence + board for", "every tracked adult"],
  },
  {
    id: "quality",
    x: 824,
    y: 244,
    lines: ["Quality report", "coverage · breaks · confidence", "attribution tier"],
  },
  {
    id: "persist",
    x: 624,
    y: 244,
    lines: ["Persist", "COPY boxes → Timescale", "result JSON → API ingest"],
  },
  // lane 3
  {
    id: "extract",
    x: 24,
    y: 354,
    lines: ["Extract", "ffmpeg → 16 kHz mono FLAC", "no -ss: audio 0 = video 0"],
  },
  {
    id: "asr",
    x: 224,
    y: 354,
    lines: ["AssemblyAI", "universal-3-5-pro · en+hi", "speakers 2–6 · words"],
    tone: "model",
  },
  {
    id: "sent",
    x: 424,
    y: 354,
    lines: ["Sentences", "from words: pause 1 s,", "punctuation, 60 w / 25 s"],
  },
  { id: "lang", x: 624, y: 354, lines: ["Language", "Hindi function words,", "not script"] },
  { id: "loud", x: 824, y: 354, lines: ["Loudness", "RMS per 0.5 s → per sentence", "(R17)"] },
  {
    id: "xlate",
    x: 1024,
    y: 354,
    lines: ["Translate", "LLM gateway (qwen 4b)", "cached by sentence"],
    tone: "model",
  },
  {
    id: "utt",
    x: 1224,
    y: 354,
    w: 230,
    lines: ["utterances", "speaker · times · text · text_en", "language · rms_db"],
    tone: "store",
  },
  // lane 4
  {
    id: "pg",
    x: 224,
    y: 466,
    w: 420,
    lines: [
      "Postgres + TimescaleDB",
      "videos · zones · tracks(meta) · events · video_analytics · utterances",
      "detection_events hypertable (compress 24 h, drop 2 d) · timetable_periods",
    ],
    tone: "store",
  },
  { id: "minio", x: 700, y: 466, lines: ["MinIO / S3", "original.mp4 · thumb.jpg"], tone: "store" },
  {
    id: "timetable",
    x: 924,
    y: 466,
    w: 230,
    lines: [
      "Timetable (Settings)",
      "per classroom · weekday · period",
      "bells · subject · teacher",
    ],
    tone: "store",
  },
  // lane 5
  {
    id: "schedule",
    x: 24,
    y: 578,
    lines: ["Period resolution", "video bells > label > overlap", "bells as ms offsets"],
  },
  {
    id: "punct",
    x: 224,
    y: 578,
    lines: ["Attendance R1–R6", "presence vs bells", "withheld when blended"],
  },
  {
    id: "prev",
    x: 424,
    y: 578,
    lines: ["Previous teacher", "handed-over adult at the bell", "left_ms · her own bell"],
  },
  {
    id: "voicerep",
    x: 624,
    y: 578,
    lines: ["Voice R17 R20–R22", "her voice by presence", "questions · languages"],
  },
  {
    id: "arc",
    x: 824,
    y: 578,
    lines: ["Lesson arc R7–R15 R18 R19", "phrases + board actions", "start = earlier signal"],
  },
  {
    id: "trust",
    x: 1024,
    y: 578,
    lines: ["Trust R22 R23", "observed / provisional /", "not observed + reason"],
  },
  {
    id: "register",
    x: 1224,
    y: 578,
    w: 230,
    lines: ["Register (lessons.day)", "own file vs spill-over", "per period, per day"],
  },
  {
    id: "dto",
    x: 424,
    y: 664,
    w: 420,
    lines: [
      "videos.get → detail DTO (oRPC, typed to the UI)",
      "lesson · punctuality · previousTeacher · voice · arc · trust · transcript · tracks · events",
    ],
  },
  // lane 6
  {
    id: "lesson",
    x: 24,
    y: 770,
    w: 400,
    lines: [
      "Lesson page",
      "At a glance · Attendance · The lesson · Voice · Board · Movement",
      "Evidence · Transcript (English) · Reliability — every time seeks the video",
    ],
    tone: "ui",
  },
  {
    id: "player",
    x: 450,
    y: 770,
    lines: ["Video player", "range stream + overlay", "boxes per track"],
    tone: "ui",
  },
  { id: "reg", x: 650, y: 770, lines: ["Register tab", "the day as periods"], tone: "ui" },
  { id: "class", x: 850, y: 770, lines: ["Classroom", "lessons · metrics · config"], tone: "ui" },
  {
    id: "settings",
    x: 1050,
    y: 770,
    w: 200,
    lines: ["Settings", "keys · timezone · GPU pod", "timetable"],
    tone: "ui",
  },
];

const EDGES: Edge[] = [
  { from: "cam", to: "upload" },
  { from: "upload", to: "store" },
  { from: "store", to: "probe" },
  { from: "probe", to: "queues" },
  { from: "queues", to: "pod", label: "POST /analyze" },
  { from: "queues", to: "fetch", via: "down", label: "video job" },
  { from: "queues", to: "extract", via: "down", label: "audio job" },
  { from: "fetch", to: "decode" },
  { from: "decode", to: "detect" },
  { from: "detect", to: "desc" },
  { from: "desc", to: "zones" },
  { from: "zones", to: "track" },
  { from: "track", to: "attr" },
  { from: "attr", to: "events", via: "down" },
  { from: "events", to: "person" },
  { from: "person", to: "quality" },
  { from: "quality", to: "persist" },
  { from: "persist", to: "pg", via: "down" },
  { from: "extract", to: "asr" },
  { from: "asr", to: "sent" },
  { from: "sent", to: "lang" },
  { from: "lang", to: "loud" },
  { from: "loud", to: "xlate" },
  { from: "xlate", to: "utt" },
  { from: "utt", to: "pg", via: "down" },
  { from: "store", to: "minio", via: "down" },
  { from: "timetable", to: "pg" },
  { from: "pg", to: "schedule", via: "down" },
  { from: "schedule", to: "punct" },
  { from: "punct", to: "prev" },
  { from: "prev", to: "voicerep" },
  { from: "voicerep", to: "arc" },
  { from: "arc", to: "trust" },
  { from: "trust", to: "register" },
  { from: "trust", to: "dto", via: "down" },
  { from: "dto", to: "lesson", via: "down" },
  { from: "dto", to: "player", via: "down" },
  { from: "register", to: "reg", via: "down" },
  { from: "minio", to: "player", via: "down", label: "stream" },
];

const TONE: Record<NonNullable<Node["tone"]>, { fill: string; stroke: string }> = {
  default: { fill: "var(--card)", stroke: "var(--border)" },
  store: {
    fill: "color-mix(in oklch, var(--tier-medium) 12%, var(--card))",
    stroke: "var(--tier-medium)",
  },
  ui: { fill: "color-mix(in oklch, var(--primary) 12%, var(--card))", stroke: "var(--primary)" },
  model: {
    fill: "color-mix(in oklch, var(--tier-high) 12%, var(--card))",
    stroke: "var(--tier-high)",
  },
};

function box(n: Node) {
  return { x: n.x, y: n.y, w: n.w ?? NODE_W, h: NODE_H };
}

function path(e: Edge, a: Node, b: Node): { d: string; lx: number; ly: number } {
  const A = box(a);
  const B = box(b);
  const sameRow = Math.abs(A.y - B.y) < 8;
  if (e.via !== "down" && sameRow) {
    const fromRight = B.x > A.x;
    const x1 = fromRight ? A.x + A.w : A.x;
    const x2 = fromRight ? B.x : B.x + B.w;
    const y = A.y + A.h / 2;
    return { d: `M ${x1} ${y} L ${x2} ${y}`, lx: (x1 + x2) / 2, ly: y - 6 };
  }
  // drop from the bottom centre of A to the top centre of B with one elbow
  const x1 = A.x + A.w / 2;
  const y1 = A.y + A.h;
  const x2 = B.x + B.w / 2;
  const y2 = B.y;
  const ym = y1 + Math.max(10, (y2 - y1) / 2);
  return {
    d: `M ${x1} ${y1} L ${x1} ${ym} L ${x2} ${ym} L ${x2} ${y2}`,
    lx: (x1 + x2) / 2,
    ly: ym - 5,
  };
}

export function ArchitectureDiagram() {
  const byId = new Map(NODES.map((n) => [n.id, n]));
  return (
    <div className="overflow-x-auto rounded-xl border border-border bg-card p-3">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto min-w-[1100px] w-full text-foreground"
        role="img"
        aria-label="End-to-end data flow: ingestion, video pipeline, audio pipeline, storage, read-time derivation, UI"
        fontFamily="ui-sans-serif, system-ui, sans-serif"
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--muted-foreground)" />
          </marker>
        </defs>

        {LANES.map((l) => (
          <g key={l.title}>
            <rect
              x={8}
              y={l.y}
              width={W - 16}
              height={l.h}
              rx={10}
              fill="color-mix(in oklch, var(--muted) 55%, transparent)"
              stroke="var(--border)"
              strokeDasharray="4 4"
            />
            <text
              x={20}
              y={l.y + 14}
              fontSize={11}
              fontWeight={600}
              fill="var(--muted-foreground)"
              letterSpacing="0.04em"
            >
              {l.title.toUpperCase()}
            </text>
          </g>
        ))}

        {EDGES.map((e, i) => {
          const a = byId.get(e.from);
          const b = byId.get(e.to);
          if (!a || !b) return null;
          const p = path(e, a, b);
          return (
            <g key={i}>
              <path
                d={p.d}
                fill="none"
                stroke="var(--muted-foreground)"
                strokeWidth={1.4}
                markerEnd="url(#arrow)"
                opacity={0.85}
              />
              {e.label && (
                <text
                  x={p.lx}
                  y={p.ly}
                  fontSize={9.5}
                  textAnchor="middle"
                  fill="var(--muted-foreground)"
                >
                  {e.label}
                </text>
              )}
            </g>
          );
        })}

        {NODES.map((n) => {
          const b = box(n);
          const tone = TONE[n.tone ?? "default"];
          return (
            <g key={n.id}>
              <rect
                x={b.x}
                y={b.y}
                width={b.w}
                height={b.h}
                rx={8}
                fill={tone.fill}
                stroke={tone.stroke}
                strokeWidth={1.2}
              />
              {n.lines.map((line, i) => (
                <text
                  key={i}
                  x={b.x + b.w / 2}
                  y={b.y + 17 + i * 13}
                  fontSize={i === 0 ? 11.5 : 9.5}
                  fontWeight={i === 0 ? 600 : 400}
                  textAnchor="middle"
                  fill={i === 0 ? "var(--foreground)" : "var(--muted-foreground)"}
                >
                  {line}
                </text>
              ))}
            </g>
          );
        })}

        <g fontSize={9.5} fill="var(--muted-foreground)">
          <rect
            x={W - 470}
            y={H - 14}
            width={10}
            height={10}
            rx={2}
            fill={TONE.model.fill}
            stroke={TONE.model.stroke}
          />
          <text x={W - 455} y={H - 5}>
            model or rule-heavy stage
          </text>
          <rect
            x={W - 300}
            y={H - 14}
            width={10}
            height={10}
            rx={2}
            fill={TONE.store.fill}
            stroke={TONE.store.stroke}
          />
          <text x={W - 285} y={H - 5}>
            storage
          </text>
          <rect
            x={W - 210}
            y={H - 14}
            width={10}
            height={10}
            rx={2}
            fill={TONE.ui.fill}
            stroke={TONE.ui.stroke}
          />
          <text x={W - 195} y={H - 5}>
            user interface
          </text>
        </g>
      </svg>
    </div>
  );
}
