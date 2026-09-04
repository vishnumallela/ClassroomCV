import {
  bigint,
  boolean,
  date,
  integer,
  jsonb,
  pgTable,
  real,
  text,
  time,
  timestamp,
  uuid,
} from "drizzle-orm/pg-core";

export type Bbox = { x: number; y: number; w: number; h: number };
export type Polygon = [number, number][];
export type ZoneMeta = { auto?: boolean; confidence?: number; method?: string };
export type EntryExitItem = { kind: string; ts_ms: number };
export type Interval = [number, number];
// Teacher dwell histogram (row-major grid_h x grid_w per-cell sample counts).
// Teacher-only since the 2026-08 KPI slimming (entry/exit, board time, heatmap).
export type Heatmap = { grid_w: number; grid_h: number; teacher: number[] };
const EMPTY_HEATMAP: Heatmap = { grid_w: 0, grid_h: 0, teacher: [] };
export type QualityTier = "high" | "medium" | "low";
// Additive per-run trust report from the ML service (services/ml-service/app/quality.py).
// Rows written before the RF-DETR pipeline carry the older identity/fragmentation
// shape; the dashboard renders whichever fields are present.
/** One tracked adult as attribution saw it (mirrors AttributionCandidateOut). */
export type AttributionCandidate = {
  track_no: number;
  first_ms: number;
  last_ms: number;
  present_ms: number;
  in_period_ms: number;
  handed_over: boolean;
  segments: number;
  /** When she left, if she was there at the bell: the end of her presence run
   *  containing it (ml attribution.LEFT_BRIDGE_MS). Absent on older rows. */
  left_ms?: number | null;
};

export type DataQuality = {
  detections: number;
  frames: number;
  sampled_frames: number;
  coverage: number;
  mean_confidence: number;
  breaks: number;
  longest_gap_ms: number;
  // Optional because this column is jsonb and rows outlive the code that wrote
  // them: a lesson analysed before the co-presence check simply has no opinion
  // on how many adults were in the room, which is NOT the same claim as "one".
  // Readers must treat absence as unknown and never as a measured single adult.
  multiple_adults_detected?: boolean;
  max_simultaneous_adults?: number;
  co_presence_ms?: number;
  attribution?: {
    confidence: QualityTier;
    reason: string;
    chosen_track_no: number | null;
    period_known: boolean;
    splits: number;
    /** Occluded crossings where the tracker had swapped two bodies; undone by appearance. */
    swaps?: number;
    candidates: AttributionCandidate[];
  } | null;
  confidence: {
    overall: QualityTier;
    coverage: QualityTier;
    continuity: QualityTier;
    teacher: QualityTier;
    attribution?: QualityTier;
  };
  notes: string[];
};

// Application settings the admin edits in the UI (Settings page): RunPod GPU
// wiring, ML service URL override. Key-value so adding a setting never needs
// a migration. Values are plain text; secrets are masked at the API layer and
// this table must never be exposed raw.
export const appSettings = pgTable("app_settings", {
  key: text("key").primaryKey(),
  value: text("value").notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

// A classroom is the unit users organize by: one physical room / camera, its
// zone configuration, and every lesson recorded in it.
export const classrooms = pgTable("classrooms", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: text("name").notNull(),
  location: text("location"),
  description: text("description"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

// Zone template for a classroom's fixed camera: seeded into every new upload's
// per-video zones, so board/door only have to be drawn once per room. Per-video
// zones stay authoritative for analysis (a bumped camera can still be fixed on
// one video without rewriting the room).
export const classroomZones = pgTable("classroom_zones", {
  id: uuid("id").primaryKey().defaultRandom(),
  classroomId: uuid("classroom_id")
    .notNull()
    .references(() => classrooms.id, { onDelete: "cascade" }),
  kind: text("kind").notNull(),
  polygon: jsonb("polygon").$type<Polygon>().notNull(),
  meta: jsonb("meta").$type<ZoneMeta | null>(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
});

export const videos = pgTable("videos", {
  id: uuid("id").primaryKey().defaultRandom(),
  // Restrict (not cascade): deleting a classroom must not silently orphan
  // object-store bytes; the API blocks deletion while videos exist.
  classroomId: uuid("classroom_id").references(() => classrooms.id, {
    onDelete: "restrict",
  }),
  title: text("title").notNull(),
  originalFilename: text("original_filename").notNull(),
  filePath: text("file_path").notNull(),
  durationMs: bigint("duration_ms", { mode: "number" }),
  fps: real("fps"),
  width: integer("width"),
  height: integer("height"),
  status: text("status").notNull().default("queued"),
  progress: real("progress").notNull().default(0),
  error: text("error"),
  // Fence token: the id of the run that currently owns this video's derived rows.
  workflowRunId: text("workflow_run_id"),
  thumbnailPath: text("thumbnail_path"),
  uploadedAt: timestamp("uploaded_at", { withTimezone: true }).notNull().defaultNow(),

  // --- Lesson details (docs/teacher-measurements.md, P1-P5) ---
  //
  // Every measurement compares a video offset (ms from the first frame) to a
  // wall-clock time from the timetable, so recordingStartedAt is the hinge
  // between the two and nothing in Group A or B resolves without it. It is
  // read from the container's creation_time at probe time (36 of 39 sample
  // recordings carry one) and stays editable for the rest.
  //
  // All nullable on purpose: an upload is never blocked on them, and a
  // measurement missing its input reports "Not Observed" rather than a guess.
  recordingStartedAt: timestamp("recording_started_at", { withTimezone: true }),
  lessonDate: date("lesson_date"),
  // Free text: schools name periods differently ("Period 3", "Block A", "II").
  // A per-classroom period table can fill this in later without a rewrite.
  period: text("period"),
  // Local wall clock, paired with lessonDate and the school timezone in
  // app_settings. Stored as time (not timestamptz) because a period is a
  // fact about the school day, not an instant, and this keeps DST out of it.
  scheduledStart: time("scheduled_start"),
  scheduledEnd: time("scheduled_end"),
  subject: text("subject"),
  yearGroup: text("year_group"),
  roomType: text("room_type"),
  hasFollowingPeriod: boolean("has_following_period"),

  // --- Audio track ---------------------------------------------------------
  //
  // Deliberately NOT `status`/`progress`. The video and audio halves run as
  // independent jobs against different services (a rented GPU, a transcription
  // API), and one status column with two writers produces a badge that flickers
  // between them. Video keeps `status`; audio reports here.
  //
  // "skipped" is a real outcome, not a failure: a lesson with no usable audio
  // still yields every Group A number.
  audioStatus: text("audio_status"),
  audioError: text("audio_error"),
  // Where the extracted 16 kHz mono FLAC lives. Kept after transcription
  // because the loudness pass (R17) reads the same file, and re-extracting it
  // from a 2 GB mp4 to retry one threshold is waste.
  audioPath: text("audio_path"),
  // The provider's transcript id. Stored so a retry resumes the existing job
  // instead of paying to transcribe the same hour of audio twice.
  transcriptId: text("transcript_id"),
});

/**
 * One turn of speech, with what the labelling pass made of it.
 *
 * The unit of storage is the utterance rather than the KPI, for the same reason
 * `detection_events` stores boxes rather than board-minutes: every number in
 * Groups C and D of docs/teacher-measurements.md is arithmetic over these rows,
 * so a changed definition is a re-derive rather than a re-spend on the API.
 *
 * Only the teacher's utterances get labelled — that decision halves both the
 * cost and the annotation surface, and it is already the documented scope.
 */
export const utterances = pgTable("utterances", {
  id: uuid("id").primaryKey().defaultRandom(),
  videoId: uuid("video_id")
    .notNull()
    .references(() => videos.id, { onDelete: "cascade" }),
  // Position in the lesson. Ordering by startMs alone is ambiguous when a
  // diarizer emits overlapping turns.
  idx: integer("idx").notNull(),
  // Whatever the diarizer called them: "teacher" when role identification
  // worked, otherwise "A"/"B"/"C".
  speaker: text("speaker").notNull(),
  // Resolved separately from `speaker`, because the diarizer's own role guess
  // is one opinion and the video is another: a turn attributed to the teacher
  // while she is out of the room is misattributed, and presence data says so.
  isTeacher: boolean("is_teacher"),
  startMs: bigint("start_ms", { mode: "number" }).notNull(),
  endMs: bigint("end_ms", { mode: "number" }).notNull(),
  text: text("text").notNull(),
  confidence: real("confidence"),
  // Per-utterance language for R21. Null until the language pass runs.
  language: text("language"),

  // --- labels (null until the labelling pass runs) -------------------------
  // instructing | explaining | asking | feedback | behaviour | procedure |
  // off_topic | closing — the set named in docs/domain-a-kpis-and-labels.md.
  intent: text("intent"),
  attentionCue: boolean("attention_cue"),
  setsTask: boolean("sets_task"),
});

export const zones = pgTable("zones", {
  id: uuid("id").primaryKey().defaultRandom(),
  videoId: uuid("video_id")
    .notNull()
    .references(() => videos.id, { onDelete: "cascade" }),
  kind: text("kind").notNull(),
  polygon: jsonb("polygon").$type<Polygon>().notNull(),
  meta: jsonb("meta").$type<ZoneMeta | null>(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
});

export const tracks = pgTable("tracks", {
  id: uuid("id").primaryKey().defaultRandom(),
  videoId: uuid("video_id")
    .notNull()
    .references(() => videos.id, { onDelete: "cascade" }),
  trackNo: integer("track_no").notNull(),
  role: text("role").notNull().default("unknown"),
  roleConfidence: real("role_confidence"),
  firstMs: bigint("first_ms", { mode: "number" }).notNull(),
  lastMs: bigint("last_ms", { mode: "number" }).notNull(),
  meta: jsonb("meta").$type<Record<string, unknown> | null>(),
});

export const events = pgTable("events", {
  id: uuid("id").primaryKey().defaultRandom(),
  videoId: uuid("video_id")
    .notNull()
    .references(() => videos.id, { onDelete: "cascade" }),
  trackNo: integer("track_no"),
  kind: text("kind").notNull(),
  videoTsMs: bigint("video_ts_ms", { mode: "number" }).notNull(),
});

// TimescaleDB hypertable, bulk-written by the ML service. No PK, no FK by design;
// deletion is handled by explicit raw SQL keyed on video_id. Partitioned on
// wall-clock ts (not per-video video_ts_ms) so compression/retention policies
// can age chunks; the ML COPY omits ts and lets the default fill it.
export const detectionEvents = pgTable("detection_events", {
  videoTsMs: bigint("video_ts_ms", { mode: "number" }).notNull(),
  videoId: uuid("video_id").notNull(),
  // Nullable since 0014. NULL = the detector called this box a teacher and no
  // tracked person owns it — a second adult in the room, or her own box on a
  // frame the chain rejected. Keeping those rows is what lets the attribution
  // rule change with a /rederive instead of another paid GPU pass; dropping
  // them is how the evidence of a two-teacher lesson was destroyed before it
  // reached the database. Only the `teacher` class is ever stored either way.
  trackNo: integer("track_no"),
  bbox: jsonb("bbox").$type<Bbox>().notNull(),
  confidence: real("confidence").notNull(),
  meta: jsonb("meta").$type<Record<string, unknown> | null>(),
  ts: timestamp("ts", { withTimezone: true }).notNull().defaultNow(),
});

export const videoAnalytics = pgTable("video_analytics", {
  videoId: uuid("video_id")
    .primaryKey()
    .references(() => videos.id, { onDelete: "cascade" }),
  teacherPresentMs: bigint("teacher_present_ms", { mode: "number" }).notNull().default(0),
  // Nullable on purpose, all three: NULL means the input was absent (no board
  // zone; or a /rederive that replayed teacher-only rows and never saw the
  // action classes), while 0 means measured and genuinely zero. Collapsing
  // them would report "she never wrote" for a lesson nobody scored.
  teacherBoardMs: bigint("teacher_board_ms", { mode: "number" }),
  teacherPointingMs: bigint("teacher_pointing_ms", { mode: "number" }),
  teacherWritingMs: bigint("teacher_writing_ms", { mode: "number" }),
  entries: integer("entries").notNull().default(0),
  exits: integer("exits").notNull().default(0),
  presenceIntervals: jsonb("presence_intervals").$type<Interval[]>().notNull().default([]),
  boardIntervals: jsonb("board_intervals").$type<Interval[]>().notNull().default([]),
  pointingIntervals: jsonb("pointing_intervals").$type<Interval[]>().notNull().default([]),
  writingIntervals: jsonb("writing_intervals").$type<Interval[]>().notNull().default([]),
  entryExit: jsonb("entry_exit").$type<EntryExitItem[]>().notNull().default([]),
  heatmap: jsonb("heatmap").$type<Heatmap>().notNull().default(EMPTY_HEATMAP),
  // Additive trust report; null for rows computed before the quality pass.
  dataQuality: jsonb("data_quality").$type<DataQuality | null>(),
  computedAt: timestamp("computed_at", { withTimezone: true }).defaultNow(),
});
