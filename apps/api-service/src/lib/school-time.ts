import { getAppSettings } from "@api/lib/app-settings";

/**
 * Turning video offsets into punctuality.
 *
 * The two halves of every measurement in docs/teacher-measurements.md speak
 * different languages. Detections are offsets: "the teacher first appears
 * 192,000 ms into the recording". The timetable is a wall clock: "period 3
 * starts at 11:15". Neither converts to the other without two more facts —
 * when the recording started (`videos.recording_started_at`, read from the
 * container's creation_time) and which timezone the school keeps.
 *
 * Everything here is pure except the settings read, so the arithmetic can be
 * tested without a database.
 */

/** Fallback when nobody has set one. India has no DST, which keeps this simple. */
export const DEFAULT_SCHOOL_TIMEZONE = "Asia/Kolkata";

export async function schoolTimezone(): Promise<string> {
  const settings = await getAppSettings().catch(() => ({}) as Record<string, never>);
  const tz = settings.schoolTimezone?.trim();
  return tz && isValidTimezone(tz) ? tz : DEFAULT_SCHOOL_TIMEZONE;
}

export function isValidTimezone(tz: string): boolean {
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: tz });
    return true;
  } catch {
    return false;
  }
}

/**
 * How far `tz` is ahead of UTC at a given instant, in ms.
 *
 * Formats the instant into the zone's own calendar fields, reads those fields
 * back as if they were UTC, and takes the difference. That is the offset the
 * zone was applying at that moment — DST included, without a date library.
 */
function zoneOffsetMs(at: Date, tz: string): number {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).formatToParts(at);

  const field = (type: string): number => Number(parts.find((p) => p.type === type)?.value ?? "0");

  // hour comes back as 24 at midnight under hour12:false in some engines.
  const asIfUtc = Date.UTC(
    field("year"),
    field("month") - 1,
    field("day"),
    field("hour") % 24,
    field("minute"),
    field("second"),
  );
  return asIfUtc - at.getTime();
}

/**
 * A local date + time in `tz`, as a UTC instant.
 *
 * Two passes: the first guesses the offset using the naive instant, the second
 * re-reads it at the corrected instant. That second pass only matters within an
 * hour of a DST boundary, and costs nothing where there is no DST.
 *
 * @param date "YYYY-MM-DD"
 * @param time "HH:MM" or "HH:MM:SS"
 */
export function schoolTimeToInstant(date: string, time: string, tz: string): Date | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return null;
  if (!/^\d{2}:\d{2}(:\d{2})?$/.test(time)) return null;

  const naive = new Date(`${date}T${time.length === 5 ? `${time}:00` : time}Z`);
  if (Number.isNaN(naive.getTime())) return null;

  const firstPass = new Date(naive.getTime() - zoneOffsetMs(naive, tz));
  return new Date(naive.getTime() - zoneOffsetMs(firstPass, tz));
}

/** The calendar date an instant falls on in `tz`, as "YYYY-MM-DD". */
export function localDateInSchoolTz(at: Date, tz: string): string {
  // en-CA formats as YYYY-MM-DD, which is the shape Postgres `date` wants.
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(at);
}

/** The wall clock an instant reads as in `tz`, as "HH:MM". */
export function localTimeInSchoolTz(at: Date, tz: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: tz,
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
  }).format(at);
}

/** A video offset in ms, as the instant it actually happened. */
export function offsetToInstant(recordingStartedAt: Date, offsetMs: number): Date {
  return new Date(recordingStartedAt.getTime() + offsetMs);
}

export interface LessonClock {
  timezone: string;
  recordingStartedAt: Date;
  /** Scheduled start as an instant, or null if the timetable is not filled in. */
  scheduledStartAt: Date | null;
  scheduledEndAt: Date | null;
}

/**
 * Everything needed to place one lesson's detections on a wall clock.
 *
 * Returns null when the recording has no anchor: without it nothing in Group A
 * or B can be computed and the honest answer is "Not Observed", not zero.
 */
export function lessonClock(
  video: {
    recordingStartedAt: Date | null;
    lessonDate: string | null;
    scheduledStart: string | null;
    scheduledEnd: string | null;
  },
  tz: string,
): LessonClock | null {
  if (!video.recordingStartedAt) return null;

  // The lesson date defaults to whatever day the recording started on, so a
  // filled-in start time still works when nobody set the date explicitly.
  const date = video.lessonDate ?? localDateInSchoolTz(video.recordingStartedAt, tz);

  return {
    timezone: tz,
    recordingStartedAt: video.recordingStartedAt,
    scheduledStartAt: video.scheduledStart
      ? schoolTimeToInstant(date, video.scheduledStart, tz)
      : null,
    scheduledEndAt: video.scheduledEnd ? schoolTimeToInstant(date, video.scheduledEnd, tz) : null,
  };
}

/**
 * Minutes between a video offset and a scheduled time. Positive is late.
 *
 * Rounded to one decimal: the underlying anchor is a container timestamp with
 * second precision at best, so reporting "3.28 minutes late" would claim a
 * confidence the input does not have.
 */
export function minutesAgainstSchedule(
  clock: LessonClock,
  offsetMs: number,
  against: Date | null,
): number | null {
  if (!against) return null;
  const actual = offsetToInstant(clock.recordingStartedAt, offsetMs);
  return Math.round(((actual.getTime() - against.getTime()) / 60_000) * 10) / 10;
}

/**
 * The scheduled period as offsets into the recording, for the ML service.
 *
 * Attribution's primary rule is about who was in the room during the PERIOD,
 * and detections are offsets from the first frame — so the period has to be
 * expressed the same way. Null when either the anchor or the timetable is
 * missing; the service then judges the whole recording and says so.
 */
export function periodOffsets(
  video: Parameters<typeof lessonClock>[0],
  tz: string,
): { startMs: number; endMs: number } | null {
  const clock = lessonClock(video, tz);
  if (!clock?.scheduledStartAt || !clock.scheduledEndAt) return null;
  const base = clock.recordingStartedAt.getTime();
  return {
    startMs: clock.scheduledStartAt.getTime() - base,
    endMs: clock.scheduledEndAt.getTime() - base,
  };
}
