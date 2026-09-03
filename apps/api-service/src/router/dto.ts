import type { getVideoDetail } from "@api/db/queries";
import {
  lessonClock,
  localTimeInSchoolTz,
  minutesAgainstSchedule,
  offsetToInstant,
} from "@api/lib/school-time";

type Detail = NonNullable<Awaited<ReturnType<typeof getVideoDetail>>>;

type Analytics = Detail["analytics"];

/**
 * Why a lesson's Group A numbers are withheld, in words a reader can act on.
 *
 * Refusing beats guessing here, and the choice is not close. The presence
 * timeline is built by following ONE person; when two adults are in the room it
 * follows whichever of them the detector scored highest on each frame, which is
 * a blend rather than a person. Every number below would then be measured
 * against the wrong body — and always in the flattering direction, because the
 * blend starts with whoever was alone first, which in a handover recording is
 * the OUTGOING teacher. A gap prompts someone to look; a confident wrong number
 * gets read as a fact.
 */
const MULTIPLE_ADULTS_REASON =
  "More than one adult was in the room during this lesson, so the teacher timeline may " +
  "blend them. Nothing yet decides which of them this lesson assesses, so arrival, " +
  "departure and time in the room are not reported.";

/**
 * Group A of docs/teacher-measurements.md, derived at read time.
 *
 * Nothing here is stored. These are three subtractions over rows that already
 * exist, so editing a lesson's scheduled start immediately corrects every
 * number that depends on it — no re-analysis, and no second copy of the truth
 * to drift out of step with the timetable.
 *
 * Every field is independently nullable, and null means "Not Observed" (R23):
 * no recording anchor, no presence data, no timetable entered, or more than one
 * adult in the room. It never means zero. `notObservedReason` is non-null only
 * for that last case — the others are visible from the inputs themselves (an
 * empty date field explains itself), while a blended timeline looks exactly
 * like a clean one and has to be said out loud.
 */
function toPunctuality(v: Detail["video"], analytics: Analytics, timezone: string) {
  const clock = lessonClock(v, timezone);
  const presence = analytics?.presenceIntervals ?? [];
  const first = presence[0];
  const last = presence.at(-1);

  // `=== true` on purpose: this field is absent on every lesson analysed before
  // the check existed, and absent means "never looked", not "one adult". Those
  // rows keep reporting their numbers rather than being retroactively withheld
  // on evidence nobody gathered.
  const blended = analytics?.dataQuality?.multiple_adults_detected === true;

  if (!clock || !first || !last || blended) {
    return {
      timezone,
      recordingStartedAt: v.recordingStartedAt?.toISOString() ?? null,
      arrivalAt: null,
      departureAt: null,
      arrivalMinutesLate: null,
      departureMinutesLate: null,
      presenceShareOfPeriod: null,
      notObservedReason: blended ? MULTIPLE_ADULTS_REASON : null,
    };
  }

  // R5's denominator is the SCHEDULED period, not the recording length: a
  // lesson recorded for 20 minutes of a 45-minute period should read 44%, not
  // 100%. That is the whole point of measuring against the timetable.
  const scheduledMs =
    clock.scheduledStartAt && clock.scheduledEndAt
      ? clock.scheduledEndAt.getTime() - clock.scheduledStartAt.getTime()
      : null;
  const presentMs = analytics?.teacherPresentMs ?? null;

  return {
    timezone,
    recordingStartedAt: v.recordingStartedAt?.toISOString() ?? null,
    arrivalAt: localTimeInSchoolTz(offsetToInstant(clock.recordingStartedAt, first[0]), timezone),
    departureAt: localTimeInSchoolTz(offsetToInstant(clock.recordingStartedAt, last[1]), timezone),
    arrivalMinutesLate: minutesAgainstSchedule(clock, first[0], clock.scheduledStartAt),
    departureMinutesLate: minutesAgainstSchedule(clock, last[1], clock.scheduledEndAt),
    presenceShareOfPeriod:
      scheduledMs !== null && scheduledMs > 0 && presentMs !== null
        ? Math.round((presentMs / scheduledMs) * 1000) / 1000
        : null,
    notObservedReason: null,
  };
}

export function toDetailDto(d: Detail, timezone: string) {
  const v = d.video;
  return {
    classroom: d.classroom ? { id: d.classroom.id, name: d.classroom.name } : null,
    video: {
      id: v.id,
      classroomId: v.classroomId,
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
    lesson: {
      recordingStartedAt: v.recordingStartedAt?.toISOString() ?? null,
      lessonDate: v.lessonDate,
      period: v.period,
      scheduledStart: v.scheduledStart,
      scheduledEnd: v.scheduledEnd,
      subject: v.subject,
      yearGroup: v.yearGroup,
      roomType: v.roomType,
      hasFollowingPeriod: v.hasFollowingPeriod,
    },
    punctuality: toPunctuality(v, d.analytics, timezone),
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
          teacherPointingMs: d.analytics.teacherPointingMs,
          teacherWritingMs: d.analytics.teacherWritingMs,
          entries: d.analytics.entries,
          exits: d.analytics.exits,
          presenceIntervals: d.analytics.presenceIntervals,
          boardIntervals: d.analytics.boardIntervals,
          pointingIntervals: d.analytics.pointingIntervals,
          writingIntervals: d.analytics.writingIntervals,
          entryExit: d.analytics.entryExit,
          heatmap: d.analytics.heatmap,
          dataQuality: d.analytics.dataQuality ?? null,
        }
      : null,
  };
}
