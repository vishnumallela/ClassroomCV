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
      dataQuality: a.data_quality ?? null,
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

/**
 * Rebuild playback frames from the PERMANENT overlay tier (RDP-simplified
 * keyframe boxes stored in tracks.meta.overlay) instead of the raw
 * detection_events hot tier. Used as the fallback when raw rows have aged out
 * under the 7-day retention policy (plan section 6): at 8h x 80-camera scale the
 * raw firehose is dropped, but the sparse ~2s-cadence keyframes are kept forever
 * at ~2% of the size, so the video overlay keeps working. Cost is O(tracks), not
 * O(detections). Returns null when a video has no overlay tier (pre-overlay rows).
 */
type OverlayKeyframe = [number, number, number, number, number]; // [ts_ms, x, y, w, h]

export async function getOverlayFrames(videoId: string): Promise<DetectionData | null> {
  const video = await getVideo(videoId);
  if (!video) return null;
  const rows = await pg<{ track_no: number; role: string; keyframes: OverlayKeyframe[] | null }[]>`
    select track_no, role, meta->'overlay'->'keyframes' as keyframes
    from tracks
    where video_id = ${videoId} and meta->'overlay'->'keyframes' is not null
    order by track_no`;

  const overlayTracks = rows
    .filter(
      (r): r is typeof r & { keyframes: OverlayKeyframe[] } =>
        Array.isArray(r.keyframes) && r.keyframes.length > 0,
    )
    .map((r) => ({
      trackNo: r.track_no,
      role: r.role,
      kfs: r.keyframes.toSorted((a, b) => a[0] - b[0]),
    }));
  if (overlayTracks.length === 0) return null;

  const roles: Record<string, string> = {};
  const times = new Set<number>();
  for (const t of overlayTracks) {
    roles[String(t.trackNo)] = t.role;
    for (const kf of t.kfs) times.add(kf[0]);
  }

  // Stride the union timeline so total emitted boxes stay under MAX_BOXES even
  // for an 8h video (which produces thousands of 2s-cadence keyframes).
  let timeline = [...times].toSorted((a, b) => a - b);
  const budget = Math.max(1, Math.floor(MAX_BOXES / overlayTracks.length));
  if (timeline.length > budget) {
    const stride = Math.ceil(timeline.length / budget);
    timeline = timeline.filter((_v, i) => i % stride === 0);
  }

  // At each kept timestamp, HOLD each active track's most-recent keyframe box
  // (binary search), so every concurrently-present person is drawn — not just
  // the few whose keyframe landed exactly on this timestamp.
  const frames = timeline
    .map((ts) => {
      const boxes: BoxTuple[] = [];
      for (const t of overlayTracks) {
        const kfs = t.kfs;
        if (ts < kfs[0]![0] || ts > kfs[kfs.length - 1]![0]) continue;
        let lo = 0;
        let hi = kfs.length - 1;
        let idx = 0;
        while (lo <= hi) {
          const mid = (lo + hi) >> 1;
          if (kfs[mid]![0] <= ts) {
            idx = mid;
            lo = mid + 1;
          } else {
            hi = mid - 1;
          }
        }
        const kf = kfs[idx]!;
        boxes.push([t.trackNo, r4(kf[1]), r4(kf[2]), r4(kf[3]), r4(kf[4])]);
      }
      return { tsMs: ts, boxes };
    })
    .filter((f) => f.boxes.length > 0);
  if (frames.length === 0) return null;

  const span = frames[frames.length - 1]!.tsMs - frames[0]!.tsMs;
  const fps = span > 0 ? Math.round(((frames.length - 1) / (span / 1000)) * 100) / 100 : 0.5;
  return {
    width: video.width ?? null,
    height: video.height ?? null,
    durationMs: video.durationMs ?? null,
    fps,
    roles,
    frames,
  };
}

export async function getDetections(videoId: string, fpsParam?: number): Promise<DetectionData> {
  const requestedFps = Math.min(30, fpsParam && fpsParam > 0 ? fpsParam : DEFAULT_FPS);
  // A non-UUID id would reach the raw stats query below and make Postgres throw
  // "invalid input syntax for type uuid" (an unhandled 500). Guard like
  // getVideo/getVideoStatus and degrade to an empty result instead.
  if (!isUuid(videoId)) {
    return { width: null, height: null, durationMs: null, fps: requestedFps, roles: {}, frames: [] };
  }

  const video = await getVideo(videoId);
  const width = video?.width ?? null;
  const height = video?.height ?? null;
  const durationMs = video?.durationMs ?? null;

  const [stats] = await pg<
    { total: number; frames: number; min_ts: number | null; max_ts: number | null }[]
  >`
    select count(*)::int as total,
           count(distinct video_ts_ms)::int as frames,
           min(video_ts_ms) as min_ts,
           max(video_ts_ms) as max_ts
    from detection_events where video_id = ${videoId}`;

  if (!stats || stats.frames === 0 || stats.total === 0) {
    // Raw hot tier is empty — either never written, or aged out by retention.
    // Fall back to the permanent overlay tier so playback overlays survive.
    const overlay = await getOverlayFrames(videoId);
    if (overlay) return overlay;
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
