import type { getVideoDetail } from "@api/db/queries";
import {
  lessonClock,
  localDateInSchoolTz,
  localTimeInSchoolTz,
  minutesAgainstSchedule,
  offsetToInstant,
  periodOffsets,
  schoolTimeToInstant,
} from "@api/lib/school-time";
import type { AttributionCandidate } from "@api/db/schema";
import { type ResolvedSchedule, clockInput, resolveSchedule } from "@api/lib/timetable";
import { lessonArc } from "@api/lib/lesson-arc";
import { trustItems } from "@api/lib/trust";
import { voiceReport } from "@api/lib/voice";

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
function toPunctuality(
  v: Detail["video"],
  analytics: Analytics,
  timezone: string,
  schedule: ResolvedSchedule,
) {
  const clock = lessonClock(clockInput(v, schedule), timezone);
  const presence = analytics?.presenceIntervals ?? [];
  const first = presence[0];
  const last = presence.at(-1);

  // `=== true` on purpose: this field is absent on every lesson analysed before
  // the check existed, and absent means "never looked", not "one adult". Those
  // rows keep reporting their numbers rather than being retroactively withheld
  // on evidence nobody gathered.
  // Phase 3 lifts the refusal only on a HIGH-confidence attribution. Medium
  // means the answer is probably right but thinly evidenced (the 6-minute
  // handover trim leaves 24 s to judge on), and a number that is probably
  // right is still the thing this card exists not to show.
  const dq = analytics?.dataQuality;
  const blended = dq?.multiple_adults_detected === true && dq?.attribution?.confidence !== "high";

  if (!clock || !first || !last || blended) {
    return {
      timezone,
      recordingStartedAt: v.recordingStartedAt?.toISOString() ?? null,
      arrivalAt: null,
      departureAt: null,
      arrivalMinutesLate: null,
      departureMinutesLate: null,
      presenceShareOfPeriod: null,
      notObservedReason: blended ? (dq?.attribution?.reason ?? MULTIPLE_ADULTS_REASON) : null,
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

/**
 * A tracked adult is "at the bell" if first seen within this long after the
 * scheduled start (or before it). Two sampled seconds beyond a first-frame miss.
 */
const AT_BELL_TOLERANCE_MS = 10_000;

/**
 * The OTHER lesson in this file: the previous period's teacher.
 *
 * A file that starts on the bell with the previous teacher still in the room
 * (docs/lesson-coverage-plan.md, "a file can cover parts of two periods") shows
 * the END of her lesson. Attribution marks her `handed_over` — present at the
 * bell, gone while the period's own teacher remained — and by the same rule
 * that picks this period's teacher (the one who stays to the end), the adult
 * in the room at the end of the previous period is that period's teacher. So
 * her last sighting here is the previous period's departure (R3), and whether
 * she stayed to her own bell is R4 against it.
 *
 * What this file cannot say: whether she was there for the WHOLE of her period.
 * That is the previous file's job (Phase C stitching). Every field is
 * independently nullable; `state` says which of the three honest answers this
 * is, and `reason` says why in words.
 *
 * What this file also cannot say: anything against HER bell. Periods at this
 * school are not back-to-back (period 2 ends 09:25, period 3 starts 09:50), so
 * "this period's start" is not "her period's end", and a file that starts on
 * this period's bell does not cover hers. Until timetable_periods exists (Phase
 * D) her departure is reported against THIS period's start — "5.6 min into the
 * period" — which is true, and nothing is claimed about her own bell.
 */
type PreviousTeacherState = "observed" | "not_observed" | "withheld" | "none";

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function noPreviousTeacher(state: Exclude<PreviousTeacherState, "observed">, reason: string) {
  return {
    state,
    reason,
    departureAt: null as string | null,
    departureMinutesIntoPeriod: null as number | null,
    adultsAtBell: 0,
    periodStart: null as string | null,
    previousPeriodEndKnown: false,
    previousPeriodLabel: null as string | null,
    previousPeriodEnd: null as string | null,
    departureMinutesAfterHerBell: null as number | null,
    breakMinutesBeforeThisPeriod: null as number | null,
    trackNo: null as number | null,
    presenceMsInThisFile: null as number | null,
    boardMsInThisFile: null as number | null,
  };
}

function toPreviousTeacher(
  v: Detail["video"],
  analytics: Analytics,
  timezone: string,
  schedule: ResolvedSchedule,
  tracks: Detail["tracks"] = [],
) {
  const clock = lessonClock(clockInput(v, schedule), timezone);
  if (!clock)
    return noPreviousTeacher(
      "not_observed",
      "No recording start time, so the bell cannot be placed in the video.",
    );
  if (!clock.scheduledStartAt)
    return noPreviousTeacher("not_observed", "No scheduled start entered for this lesson.");
  const bellMs = clock.scheduledStartAt.getTime() - clock.recordingStartedAt.getTime();
  const bellLocal = localTimeInSchoolTz(clock.scheduledStartAt, timezone);
  if (bellMs < 0) {
    return noPreviousTeacher(
      "not_observed",
      `The recording starts after the ${bellLocal} bell, so whether anyone was still in the room at it is not covered.`,
    );
  }
  const attribution = analytics?.dataQuality?.attribution;
  if (!attribution)
    return noPreviousTeacher(
      "not_observed",
      "Analysed before attribution existed; re-analyse to see who was in the room at the bell.",
    );
  if (attribution.confidence === "low") return noPreviousTeacher("withheld", attribution.reason);

  const atBell = (attribution.candidates ?? []).filter(
    (c) => c.handed_over && c.first_ms <= bellMs + AT_BELL_TOLERANCE_MS,
  );
  if (atBell.length === 0) {
    return noPreviousTeacher("none", `No other adult was in the room at the ${bellLocal} bell.`);
  }
  // Several handed-over adults at the bell is usually one person tracked in
  // pieces (a seated teacher's head-only lane); the last to leave is the one
  // whose departure is the lesson's. Say how many, so a reader can judge.
  //
  // "Left" is the end of her presence run containing the bell (left_ms), not
  // her last sighting: on the full handover recording a colleague in a white
  // shirt who came in 34 minutes later linked to the cream-striped period-2
  // teacher by appearance, and her last sighting would have put the period-2
  // teacher's departure at 10:32 instead of 09:55. Rows analysed before
  // left_ms existed fall back to the last sighting.
  const leftMs = (c: AttributionCandidate) => c.left_ms ?? c.last_ms;
  const previous = atBell.reduce((a, b) => (leftMs(b) > leftMs(a) ? b : a));
  const herTrack = tracks.find((t) => t.trackNo === previous.track_no);
  const departureAt = offsetToInstant(clock.recordingStartedAt, leftMs(previous));
  const minutesAfterBell = Math.round(((leftMs(previous) - bellMs) / 60_000) * 10) / 10;

  // Her OWN bell, when the classroom's timetable knows the period before this
  // one. Periods at this school are not back-to-back (period 2 ends 09:25,
  // period 3 starts 09:50), so the two numbers differ by the break: "5.6 min
  // into this period" is also "30.6 min after her bell, 25 of them break".
  const date = v.lessonDate ?? localDateInSchoolTz(clock.recordingStartedAt, timezone);
  const herEnd = schedule.previousPeriod
    ? schoolTimeToInstant(date, schedule.previousPeriod.scheduledEnd, timezone)
    : null;
  const minutesAfterHerBell = herEnd
    ? Math.round(((departureAt.getTime() - herEnd.getTime()) / 60_000) * 10) / 10
    : null;
  const breakMinutes = herEnd
    ? Math.round(((clock.scheduledStartAt.getTime() - herEnd.getTime()) / 60_000) * 10) / 10
    : null;
  const herBellText =
    herEnd && schedule.previousPeriod && minutesAfterHerBell !== null
      ? ` Her own period (${schedule.previousPeriod.label}) ended at ` +
        `${localTimeInSchoolTz(herEnd, timezone)}; she left ${minutesAfterHerBell} min after it` +
        (breakMinutes && breakMinutes > 0
          ? `, ${breakMinutes} min of which was the break before this period.`
          : ".")
      : "";
  return {
    state: "observed" as const,
    reason:
      `An adult was in the room at the ${bellLocal} bell and left at ` +
      `${localTimeInSchoolTz(departureAt, timezone)} while this period's teacher remained` +
      (atBell.length > 1
        ? ` (${atBell.length} such adults were tracked; the last to leave is taken).`
        : ".") +
      herBellText,
    departureAt: localTimeInSchoolTz(departureAt, timezone),
    departureMinutesIntoPeriod: minutesAfterBell,
    adultsAtBell: atBell.length,
    periodStart: bellLocal,
    previousPeriodEndKnown: herEnd !== null,
    previousPeriodLabel: schedule.previousPeriod?.label ?? null,
    previousPeriodEnd: herEnd ? localTimeInSchoolTz(herEnd, timezone) : null,
    departureMinutesAfterHerBell: minutesAfterHerBell,
    breakMinutesBeforeThisPeriod: breakMinutes,
    // Her own minutes in THIS file — presence and board time by the same
    // rules as the teacher's — so her period can be credited with them.
    trackNo: previous.track_no,
    presenceMsInThisFile: numberOrNull(herTrack?.meta?.present_ms),
    boardMsInThisFile: numberOrNull(herTrack?.meta?.board_ms),
  };
}

export function toDetailDto(d: Detail, timezone: string) {
  const v = d.video;
  // The lesson placed in its classroom's week: the video's own bells win,
  // else the timetable's row for its period (lib/timetable.ts).
  const schedule = resolveSchedule(v, d.timetable ?? [], timezone);
  // Group D. The teacher's voice is the one that carries the speech while the
  // video says she was in the room, so a blended or missing presence timeline
  // withholds the split rather than guessing (lib/voice.ts).
  const dq = d.analytics?.dataQuality;
  const blended = dq?.multiple_adults_detected === true && dq?.attribution?.confidence !== "high";
  const utterances = d.utterances ?? [];
  const voice = voiceReport({
    utterances: utterances.map((u) => ({
      idx: u.idx,
      speaker: u.speaker,
      startMs: u.startMs,
      endMs: u.endMs,
      text: u.text,
      confidence: u.confidence,
      language: u.language,
      rmsDb: u.rmsDb,
      textEn: u.textEn,
    })),
    durationMs: v.durationMs,
    presenceIntervals:
      d.analytics && !blended ? (d.analytics.presenceIntervals as [number, number][]) : null,
    audioStatus: v.audioStatus,
    audioError: v.audioError,
  });
  // Groups B and C: when the lesson started and ended, and how it ended — the
  // teacher's sentences against the video's board and presence intervals and
  // the bells (lib/lesson-arc.ts).
  const presence =
    d.analytics && !blended ? (d.analytics.presenceIntervals as [number, number][]) : null;
  const bells = periodOffsets(clockInput(v, schedule), timezone);
  const arc = lessonArc({
    sentences: voice.teacher.speaker
      ? utterances
          .filter((u) => u.speaker === voice.teacher.speaker)
          .map((u) => ({
            idx: u.idx,
            startMs: u.startMs,
            endMs: u.endMs,
            text: u.text,
            textEn: u.textEn,
            language: u.language,
          }))
      : [],
    noSentencesReason:
      utterances.length > 0 && !voice.teacher.speaker
        ? `The teacher's voice could not be told apart: ${voice.teacher.reason}`
        : undefined,
    boardIntervals: (d.analytics?.boardIntervals as [number, number][] | undefined) ?? [],
    actionIntervals: [
      ...((d.analytics?.pointingIntervals as [number, number][] | undefined) ?? []),
      ...((d.analytics?.writingIntervals as [number, number][] | undefined) ?? []),
    ],
    durationMs: v.durationMs ?? 0,
    bellStartMs: bells?.startMs ?? null,
    bellEndMs: bells?.endMs ?? null,
    arrivalMs: presence?.[0]?.[0] ?? null,
    departureMs: presence?.[presence.length - 1]?.[1] ?? null,
  });
  const punctuality = toPunctuality(v, d.analytics, timezone, schedule);
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
      audioStatus: v.audioStatus,
      audioError: v.audioError,
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
      // What the numbers were actually measured against, and where it came from.
      schedule: {
        weekday: schedule.weekday,
        period: schedule.period,
        scheduledStart: schedule.scheduledStart,
        scheduledEnd: schedule.scheduledEnd,
        subject: schedule.subject,
        yearGroup: schedule.yearGroup,
        teacher: schedule.teacher,
        hasFollowingPeriod: schedule.hasFollowingPeriod,
        previousPeriod: schedule.previousPeriod,
        source: schedule.source,
      },
    },
    voice,
    transcript: utterances.map((u) => ({
      idx: u.idx,
      speaker: u.speaker,
      isTeacher: voice.teacher.speaker ? u.speaker === voice.teacher.speaker : null,
      startMs: u.startMs,
      endMs: u.endMs,
      text: u.text,
      textEn: u.textEn,
      confidence: u.confidence,
      language: u.language,
      intent: u.intent,
      attentionCue: u.attentionCue,
      setsTask: u.setsTask,
    })),
    punctuality,
    previousTeacher: toPreviousTeacher(v, d.analytics, timezone, schedule, d.tracks),
    arc,
    // R22 and R23 over every measurement: what was observed, what is
    // provisional, and what is Not Observed and why.
    trust: trustItems({
      punctuality,
      videoAnalysed: d.analytics !== null && !blended,
      videoQuality: dq?.confidence?.overall ?? null,
      videoCoverage: dq?.coverage ?? null,
      arc,
      voice,
    }),
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
