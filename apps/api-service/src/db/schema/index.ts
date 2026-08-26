import { bigint, integer, jsonb, pgTable, real, text, timestamp, uuid } from "drizzle-orm/pg-core";

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
export type DataQuality = {
  detections: number;
  frames: number;
  sampled_frames: number;
  coverage: number;
  mean_confidence: number;
  breaks: number;
  longest_gap_ms: number;
  confidence: {
    overall: QualityTier;
    coverage: QualityTier;
    continuity: QualityTier;
    teacher: QualityTier;
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
  trackNo: integer("track_no").notNull(),
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
