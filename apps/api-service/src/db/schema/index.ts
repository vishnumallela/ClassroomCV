import { bigint, integer, jsonb, pgTable, real, text, timestamp, uuid } from "drizzle-orm/pg-core";

export type Bbox = { x: number; y: number; w: number; h: number };
export type Polygon = [number, number][];
export type ZoneMeta = { auto?: boolean; confidence?: number; method?: string };
export type OccupancyPoint = { ts_ms: number; students: number; teacher: boolean };
export type EntryExitItem = { kind: string; ts_ms: number };
export type Interval = [number, number];
// Spatial dwell histograms (row-major grid_h x grid_w per-cell sample counts).
export type Heatmap = { grid_w: number; grid_h: number; teacher: number[]; students: number[] };
const EMPTY_HEATMAP: Heatmap = { grid_w: 0, grid_h: 0, teacher: [], students: [] };
export type QualityTier = "high" | "medium" | "low";
// Additive per-run trust report from the ML service (services/ml-service/app/quality.py).
export type DataQuality = {
  detections: number;
  frames: number;
  identities: number;
  raw_tracks: number;
  fragmentation: number;
  coverage: number;
  occupied_buckets: number;
  span_buckets: number;
  concurrent_peak: number;
  concurrent_typical: number;
  confidence: {
    overall: QualityTier;
    occupancy: QualityTier;
    identity: QualityTier;
    coverage: QualityTier;
    teacher: QualityTier;
  };
  notes: string[];
};
// One debounced teacher board-interaction segment (pointing/writing/near).
export type BoardInteraction = { kind: "pointing" | "writing" | "near"; start_ms: number; end_ms: number };

export const videos = pgTable("videos", {
  id: uuid("id").primaryKey().defaultRandom(),
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
  teacherBoardMs: bigint("teacher_board_ms", { mode: "number" }),
  entries: integer("entries").notNull().default(0),
  exits: integer("exits").notNull().default(0),
  avgStudents: real("avg_students"),
  maxStudents: integer("max_students"),
  presenceIntervals: jsonb("presence_intervals").$type<Interval[]>().notNull().default([]),
  boardIntervals: jsonb("board_intervals").$type<Interval[]>().notNull().default([]),
  entryExit: jsonb("entry_exit").$type<EntryExitItem[]>().notNull().default([]),
  occupancy: jsonb("occupancy").$type<OccupancyPoint[]>().notNull().default([]),
  heatmap: jsonb("heatmap").$type<Heatmap>().notNull().default(EMPTY_HEATMAP),
  // Teacher board-interaction analytics (null ms when no board zone exists).
  teacherPointingMs: bigint("teacher_pointing_ms", { mode: "number" }),
  teacherWritingMs: bigint("teacher_writing_ms", { mode: "number" }),
  teacherBoardNearMs: bigint("teacher_board_near_ms", { mode: "number" }),
  boardInteractions: jsonb("board_interactions").$type<BoardInteraction[]>().notNull().default([]),
  // Additive trust report; null for rows computed before the quality pass.
  dataQuality: jsonb("data_quality").$type<DataQuality | null>(),
  computedAt: timestamp("computed_at", { withTimezone: true }).defaultNow(),
});
