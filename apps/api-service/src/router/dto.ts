import type { getVideoDetail } from "@api/db/queries";

type Detail = NonNullable<Awaited<ReturnType<typeof getVideoDetail>>>;

export function toDetailDto(d: Detail) {
  const v = d.video;
  return {
    video: {
      id: v.id,
      title: v.title,
      originalFilename: v.originalFilename,
      durationMs: v.durationMs,
      fps: v.fps,
      width: v.width,
      height: v.height,
      status: v.status,
      progress: v.progress,
      error: v.error,
      thumbnailUrl: v.thumbnailPath ? `/videos/${v.id}/thumbnail` : null,
      uploadedAt: v.uploadedAt.toISOString(),
    },
    zones: d.zones.map((z) => ({
      id: z.id,
      kind: z.kind,
      polygon: z.polygon,
      meta: z.meta ?? null,
    })),
    tracks: d.tracks.map((t) => ({
      trackNo: t.trackNo,
      role: t.role,
      roleConfidence: t.roleConfidence,
      firstMs: t.firstMs,
      lastMs: t.lastMs,
    })),
    events: d.events.map((e) => ({ kind: e.kind, videoTsMs: e.videoTsMs, trackNo: e.trackNo })),
    analytics: d.analytics
      ? {
          teacherPresentMs: d.analytics.teacherPresentMs,
          teacherBoardMs: d.analytics.teacherBoardMs,
          entries: d.analytics.entries,
          exits: d.analytics.exits,
          avgStudents: d.analytics.avgStudents,
          maxStudents: d.analytics.maxStudents,
          presenceIntervals: d.analytics.presenceIntervals,
          boardIntervals: d.analytics.boardIntervals,
          entryExit: d.analytics.entryExit,
          occupancy: d.analytics.occupancy,
          heatmap: d.analytics.heatmap,
          teacherPointingMs: d.analytics.teacherPointingMs,
          teacherWritingMs: d.analytics.teacherWritingMs,
          teacherBoardNearMs: d.analytics.teacherBoardNearMs,
          boardInteractions: d.analytics.boardInteractions,
        }
      : null,
  };
}
