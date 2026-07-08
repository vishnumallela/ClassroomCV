import { and, desc, eq, sql } from "drizzle-orm";
import { db, sql as pg } from "@api/lib/db";
import {
  type Bbox,
  events,
  type Polygon,
  tracks,
  videoAnalytics,
  videos,
  type ZoneMeta,
  zones,
} from "@api/db/schema";
import type { AnalysisResult } from "@api/lib/ml";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export function isUuid(value: string): boolean {
  return UUID_RE.test(value);
}

function chunk<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

export type VideoRow = typeof videos.$inferSelect;
export type ZoneInput = { kind: string; polygon: Polygon };

export async function createVideo(input: {
  id: string;
  title: string;
  originalFilename: string;
  filePath: string;
}): Promise<VideoRow> {
  const [row] = await db
    .insert(videos)
    .values({
      id: input.id,
      title: input.title,
      originalFilename: input.originalFilename,
      filePath: input.filePath,
      status: "queued",
      progress: 0,
    })
    .returning();
  return row!;
}

export async function getVideo(id: string): Promise<VideoRow | undefined> {
  if (!isUuid(id)) return undefined;
  const [row] = await db.select().from(videos).where(eq(videos.id, id));
  return row;
}

export async function updateVideo(
  id: string,
  patch: Partial<typeof videos.$inferInsert>,
): Promise<void> {
  await db.update(videos).set(patch).where(eq(videos.id, id));
}

export async function updateStatus(
  id: string,
  patch: { status?: string; progress?: number; error?: string | null },
): Promise<void> {
  await db.update(videos).set(patch).where(eq(videos.id, id));
}

export async function setWorkflowRunId(id: string, workflowRunId: string): Promise<void> {
  await db.update(videos).set({ workflowRunId }).where(eq(videos.id, id));
}

export interface VideoListItem {
  id: string;
  title: string;
  status: string;
  progress: number;
  durationMs: number | null;
  uploadedAt: string;
  thumbnailUrl: string | null;
  error: string | null;
  teacherPresentMs: number | null;
  entries: number | null;
  exits: number | null;
}

export async function listVideos(): Promise<VideoListItem[]> {
  const rows = await db
    .select({
      id: videos.id,
      title: videos.title,
      status: videos.status,
      progress: videos.progress,
      durationMs: videos.durationMs,
      uploadedAt: videos.uploadedAt,
      thumbnailPath: videos.thumbnailPath,
      error: videos.error,
      teacherPresentMs: videoAnalytics.teacherPresentMs,
      entries: videoAnalytics.entries,
      exits: videoAnalytics.exits,
    })
    .from(videos)
    .leftJoin(videoAnalytics, eq(videoAnalytics.videoId, videos.id))
    .orderBy(desc(videos.uploadedAt));

  return rows.map((r) => {
    const done = r.status === "done";
    return {
      id: r.id,
      title: r.title,
      status: r.status,
      progress: r.progress,
      durationMs: r.durationMs,
      uploadedAt: r.uploadedAt.toISOString(),
      thumbnailUrl: r.thumbnailPath ? `/videos/${r.id}/thumbnail` : null,
      error: r.error,
      teacherPresentMs: done ? r.teacherPresentMs : null,
      entries: done ? r.entries : null,
      exits: done ? r.exits : null,
    };
  });
}

export async function getVideoDetail(id: string) {
  const video = await getVideo(id);
  if (!video) return undefined;
  const [zoneRows, trackRows, eventRows, analyticsRows] = await Promise.all([
    db.select().from(zones).where(eq(zones.videoId, id)).orderBy(zones.createdAt),
    db.select().from(tracks).where(eq(tracks.videoId, id)).orderBy(tracks.trackNo),
    db.select().from(events).where(eq(events.videoId, id)).orderBy(events.videoTsMs),
    db.select().from(videoAnalytics).where(eq(videoAnalytics.videoId, id)),
  ]);
  return {
    video,
    zones: zoneRows,
    tracks: trackRows,
    events: eventRows,
    analytics: analyticsRows[0] ?? null,
  };
}

export async function getVideoStatus(id: string) {
  if (!isUuid(id)) return undefined;
  const [row] = await db
    .select({ status: videos.status, progress: videos.progress, error: videos.error })
    .from(videos)
    .where(eq(videos.id, id));
  return row;
}

export async function getZones(videoId: string): Promise<ZoneInput[]> {
  return db
    .select({ kind: zones.kind, polygon: zones.polygon })
    .from(zones)
    .where(eq(zones.videoId, videoId))
    .orderBy(zones.createdAt);
}

export async function hasZoneKind(videoId: string, kind: string): Promise<boolean> {
  const [row] = await db
    .select({ id: zones.id })
    .from(zones)
    .where(and(eq(zones.videoId, videoId), eq(zones.kind, kind)))
    .limit(1);
  return row !== undefined;
}

export async function insertZone(
  videoId: string,
  zone: { kind: string; polygon: Polygon; meta?: ZoneMeta | null },
): Promise<void> {
  await db.insert(zones).values({
    videoId,
    kind: zone.kind,
    polygon: zone.polygon,
    meta: zone.meta ?? null,
  });
}

export async function replaceZones(videoId: string, newZones: ZoneInput[]): Promise<void> {
  await db.transaction(async (tx) => {
    await tx.delete(zones).where(eq(zones.videoId, videoId));
    if (newZones.length > 0) {
      await tx
        .insert(zones)
        .values(newZones.map((z) => ({ videoId, kind: z.kind, polygon: z.polygon, meta: null })));
    }
  });
}

export async function countDetectionEvents(videoId: string): Promise<number> {
  const rows = await pg<{ count: number }[]>`
    select count(*)::int as count from detection_events where video_id = ${videoId}`;
  return rows[0]?.count ?? 0;
}

export async function replaceDerived(
  videoId: string,
  result: AnalysisResult,
  opts: { markDone?: boolean } = {},
): Promise<void> {
  await db.transaction(async (tx) => {
    await tx.delete(tracks).where(eq(tracks.videoId, videoId));
    await tx.delete(events).where(eq(events.videoId, videoId));
    await tx.delete(videoAnalytics).where(eq(videoAnalytics.videoId, videoId));

    const trackValues = result.tracks.map((t) => ({
      videoId,
      trackNo: t.track_no,
      role: t.role ?? "unknown",
      roleConfidence: t.role_confidence,
      firstMs: t.first_ms,
      lastMs: t.last_ms,
      meta: t.meta ?? null,
    }));
    for (const batch of chunk(trackValues, 1000)) await tx.insert(tracks).values(batch);

    const eventValues = result.events.map((e) => ({
      videoId,
      trackNo: e.track_no,
      kind: e.kind,
      videoTsMs: e.video_ts_ms,
    }));
    for (const batch of chunk(eventValues, 1000)) await tx.insert(events).values(batch);

    const a = result.analytics;
    await tx.insert(videoAnalytics).values({
      videoId,
      teacherPresentMs: a.teacher_present_ms ?? 0,
      teacherBoardMs: a.teacher_board_ms,
      entries: a.entries ?? 0,
      exits: a.exits ?? 0,
      avgStudents: a.avg_students,
      maxStudents: a.max_students,
      presenceIntervals: a.presence_intervals ?? [],
      boardIntervals: a.board_intervals ?? [],
      entryExit: a.entry_exit ?? [],
      occupancy: a.occupancy ?? [],
      heatmap: a.heatmap ?? { grid_w: 0, grid_h: 0, teacher: [], students: [] },
      teacherPointingMs: a.teacher_pointing_ms ?? null,
      teacherWritingMs: a.teacher_writing_ms ?? null,
      teacherBoardNearMs: a.teacher_board_near_ms ?? null,
      boardInteractions: a.board_interactions ?? [],
      computedAt: new Date(),
    });

    if (opts.markDone) {
      await tx
        .update(videos)
        .set({ status: "done", progress: 1, error: null })
        .where(eq(videos.id, videoId));
    }
  });
}

export async function wipeDerived(videoId: string): Promise<void> {
  await db.transaction(async (tx) => {
    await tx.delete(tracks).where(eq(tracks.videoId, videoId));
    await tx.delete(events).where(eq(events.videoId, videoId));
    await tx.delete(videoAnalytics).where(eq(videoAnalytics.videoId, videoId));
  });
}

export async function deleteVideoRows(id: string): Promise<void> {
  // detection_events has no FK, so it is deleted explicitly; both in one tx so a
  // crash cannot strand a done video whose raw detections are already gone.
  await db.transaction(async (tx) => {
    await tx.execute(sql`delete from detection_events where video_id = ${id}`);
    await tx.delete(videos).where(eq(videos.id, id));
  });
}

const MAX_BOXES = 55_000;
const DEFAULT_FPS = 5;
function r4(n: number): number {
  return Math.round(n * 1e4) / 1e4;
}

export type BoxTuple = [number, number, number, number, number];
export interface DetectionData {
  width: number | null;
  height: number | null;
  durationMs: number | null;
  fps: number;
  roles: Record<string, string>;
  frames: { tsMs: number; boxes: BoxTuple[] }[];
}

export async function getDetections(videoId: string, fpsParam?: number): Promise<DetectionData> {
  const video = await getVideo(videoId);
  const width = video?.width ?? null;
  const height = video?.height ?? null;
  const durationMs = video?.durationMs ?? null;
  const requestedFps = Math.min(30, fpsParam && fpsParam > 0 ? fpsParam : DEFAULT_FPS);

  const [stats] = await pg<
    { total: number; frames: number; min_ts: number | null; max_ts: number | null }[]
  >`
    select count(*)::int as total,
           count(distinct video_ts_ms)::int as frames,
           min(video_ts_ms) as min_ts,
           max(video_ts_ms) as max_ts
    from detection_events where video_id = ${videoId}`;

  if (!stats || stats.frames === 0 || stats.total === 0) {
    return { width, height, durationMs, fps: requestedFps, roles: {}, frames: [] };
  }

  const roleRows = await pg<{ track_no: number; role: string }[]>`
    select track_no, role from tracks where video_id = ${videoId}`;
  const roles: Record<string, string> = {};
  for (const r of roleRows) roles[String(r.track_no)] = r.role;

  const frames = stats.frames;
  const spanMs = Number(stats.max_ts ?? 0) - Number(stats.min_ts ?? 0);
  const storedFps = spanMs > 0 ? (frames - 1) / (spanMs / 1000) : requestedFps;
  let stride = Math.max(1, Math.round(storedFps / requestedFps));
  const avgBoxesPerFrame = stats.total / frames;
  const estBoxes = Math.ceil(frames / stride) * avgBoxesPerFrame;
  if (estBoxes > MAX_BOXES) stride *= Math.ceil(estBoxes / MAX_BOXES);
  const effectiveFps = Math.round((storedFps / stride) * 100) / 100;

  const rows = await pg<{ video_ts_ms: number; track_no: number; bbox: Bbox }[]>`
    with distinct_ts as (
      select video_ts_ms, row_number() over (order by video_ts_ms) as rn
      from (select distinct video_ts_ms from detection_events where video_id = ${videoId}) t
    ),
    kept as (select video_ts_ms from distinct_ts where rn % ${stride} = 0)
    select d.video_ts_ms, d.track_no, d.bbox
    from detection_events d
    join kept k on k.video_ts_ms = d.video_ts_ms
    where d.video_id = ${videoId}
    order by d.video_ts_ms, d.track_no`;

  const frameMap = new Map<number, BoxTuple[]>();
  for (const row of rows) {
    const ts = Number(row.video_ts_ms);
    let boxes = frameMap.get(ts);
    if (!boxes) {
      boxes = [];
      frameMap.set(ts, boxes);
    }
    boxes.push([row.track_no, r4(row.bbox.x), r4(row.bbox.y), r4(row.bbox.w), r4(row.bbox.h)]);
  }
  const outFrames = [...frameMap.entries()]
    .toSorted((a, b) => a[0] - b[0])
    .map(([tsMs, boxes]) => ({ tsMs, boxes }));

  return { width, height, durationMs, fps: effectiveFps, roles, frames: outFrames };
}
