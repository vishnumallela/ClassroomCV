import { getTimetable, type VideoRow } from "@api/db/queries";
import {
  type LessonClock,
  lessonClock,
  localDateInSchoolTz,
  periodOffsets,
  schoolTimeToInstant,
} from "@api/lib/school-time";

/**
 * Placing a lesson in its classroom's week (docs/lesson-coverage-plan.md,
 * Phase D).
 *
 * A lesson is a period on the classroom's wall-clock day; a file is coverage
 * of it. Until this module the bell times were typed on every video. Now the
 * classroom's timetable supplies them, and the video's own columns are an
 * OVERRIDE — set when someone corrects one lesson, or on lessons recorded
 * before the table existed — that wins whenever both are present.
 *
 * Which period a file belongs to is decided in this order:
 *   1. the period label typed on the video, matched against the day's rows;
 *   2. failing that, the row whose window the recording overlaps most,
 *      which needs the recording's anchor and duration;
 *   3. failing that, nothing — the numbers report Not Observed as before.
 *
 * Everything here is pure except `periodOffsetsFor`, so the placement can be
 * tested without a database.
 */

export interface TimetableRow {
  weekday: number;
  slot: number;
  label: string;
  scheduledStart: string;
  scheduledEnd: string;
  subject: string | null;
  teacher: string | null;
  yearGroup: string | null;
}

export interface ScheduleVideo {
  recordingStartedAt: Date | null;
  durationMs: number | null;
  lessonDate: string | null;
  period: string | null;
  scheduledStart: string | null;
  scheduledEnd: string | null;
  subject: string | null;
  yearGroup: string | null;
  hasFollowingPeriod: boolean | null;
}

export interface ResolvedSchedule {
  /** ISO weekday of the lesson (1 = Monday), or null without a date. */
  weekday: number | null;
  period: string | null;
  scheduledStart: string | null;
  scheduledEnd: string | null;
  subject: string | null;
  yearGroup: string | null;
  teacher: string | null;
  hasFollowingPeriod: boolean | null;
  /** The period that ends before this one starts, from the timetable. */
  previousPeriod: { label: string; scheduledEnd: string } | null;
  /** Where the bell times came from. */
  source: "video" | "timetable" | null;
}

/** ISO weekday (1 = Monday … 7 = Sunday) of a "YYYY-MM-DD" calendar date. */
export function isoWeekday(date: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return null;
  const d = new Date(`${date}T12:00:00Z`);
  if (Number.isNaN(d.getTime())) return null;
  const sundayFirst = d.getUTCDay();
  return sundayFirst === 0 ? 7 : sundayFirst;
}

/** "9:50" / "09:50" / "09:50:00" → "09:50:00", for comparisons. */
export function normaliseTime(value: string | null): string | null {
  if (!value) return null;
  const m = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/.exec(value.trim());
  if (!m) return null;
  const [, hh = "", mm = "", ss] = m;
  return `${hh.padStart(2, "0")}:${mm}:${ss ?? "00"}`;
}

function normaliseLabel(label: string): string {
  return label.trim().toLowerCase().replace(/\s+/g, " ");
}

function digitsOf(label: string): string | null {
  const m = /\d+/.exec(label);
  return m ? String(Number(m[0])) : null;
}

/**
 * The day's row named by the video's period label. "Period 3", "period 3",
 * "P3" and "3" all name the same row, because schools abbreviate freely and a
 * label typed on a video must not miss the table over a space.
 */
function matchByLabel(rows: TimetableRow[], label: string): TimetableRow | null {
  const wanted = normaliseLabel(label);
  const exact = rows.find((r) => normaliseLabel(r.label) === wanted);
  if (exact) return exact;
  const digits = digitsOf(label);
  if (digits === null) return null;
  const byNumber = rows.filter((r) => digitsOf(r.label) === digits);
  return byNumber.length === 1 ? (byNumber[0] ?? null) : null;
}

/** The day's row the recording's wall-clock window overlaps most. */
function matchByOverlap(
  rows: TimetableRow[],
  date: string,
  recordingStartedAt: Date,
  durationMs: number,
  tz: string,
): TimetableRow | null {
  const from = recordingStartedAt.getTime();
  const to = from + durationMs;
  let best: { row: TimetableRow; overlap: number } | null = null;
  for (const row of rows) {
    const start = schoolTimeToInstant(date, row.scheduledStart, tz);
    const end = schoolTimeToInstant(date, row.scheduledEnd, tz);
    if (!start || !end) continue;
    const overlap = Math.min(to, end.getTime()) - Math.max(from, start.getTime());
    if (overlap > 0 && (best === null || overlap > best.overlap)) best = { row, overlap };
  }
  return best ? best.row : null;
}

export function resolveSchedule(
  video: ScheduleVideo,
  timetable: TimetableRow[],
  tz: string,
): ResolvedSchedule {
  const date =
    video.lessonDate ??
    (video.recordingStartedAt ? localDateInSchoolTz(video.recordingStartedAt, tz) : null);
  const weekday = date ? isoWeekday(date) : null;
  const day = weekday
    ? timetable.filter((r) => r.weekday === weekday).sort((a, b) => a.slot - b.slot)
    : [];

  let match: TimetableRow | null = null;
  if (day.length > 0) {
    if (video.period) match = matchByLabel(day, video.period);
    if (!match && date && video.recordingStartedAt && video.durationMs) {
      match = matchByOverlap(day, date, video.recordingStartedAt, video.durationMs, tz);
    }
  }

  const explicitStart = normaliseTime(video.scheduledStart);
  const explicitEnd = normaliseTime(video.scheduledEnd);
  const explicit = explicitStart !== null && explicitEnd !== null;
  const source: ResolvedSchedule["source"] = explicit ? "video" : match ? "timetable" : null;

  const start = explicit ? explicitStart : match ? normaliseTime(match.scheduledStart) : null;
  const end = explicit ? explicitEnd : match ? normaliseTime(match.scheduledEnd) : null;

  // A following period exists when some row on the day starts exactly where
  // this one ends; a break is a gap. Only the timetable can say so.
  const follows =
    match && end ? day.some((r) => r !== match && normaliseTime(r.scheduledStart) === end) : null;
  const previous =
    match && start
      ? day
          .filter((r) => r !== match && (normaliseTime(r.scheduledEnd) ?? "") <= start)
          .sort((a, b) =>
            (normaliseTime(b.scheduledEnd) ?? "").localeCompare(
              normaliseTime(a.scheduledEnd) ?? "",
            ),
          )[0]
      : undefined;

  return {
    weekday,
    period: video.period ?? match?.label ?? null,
    scheduledStart: start,
    scheduledEnd: end,
    subject: video.subject ?? match?.subject ?? null,
    yearGroup: video.yearGroup ?? match?.yearGroup ?? null,
    teacher: match?.teacher ?? null,
    hasFollowingPeriod: video.hasFollowingPeriod ?? follows,
    previousPeriod: previous
      ? {
          label: previous.label,
          scheduledEnd: normaliseTime(previous.scheduledEnd) ?? previous.scheduledEnd,
        }
      : null,
    source,
  };
}

/** The video-shaped input `lessonClock` wants, with the resolved bells. */
export function clockInput(video: ScheduleVideo, schedule: ResolvedSchedule) {
  return {
    recordingStartedAt: video.recordingStartedAt,
    lessonDate: video.lessonDate,
    scheduledStart: schedule.scheduledStart,
    scheduledEnd: schedule.scheduledEnd,
  };
}

export function resolvedClock(
  video: ScheduleVideo,
  schedule: ResolvedSchedule,
  tz: string,
): LessonClock | null {
  return lessonClock(clockInput(video, schedule), tz);
}

/**
 * The scheduled period as offsets into the recording, for the ML service —
 * from the classroom's timetable when the video has no bells of its own.
 */
export async function periodOffsetsFor(
  video: VideoRow,
  tz: string,
): Promise<{ startMs: number; endMs: number } | null> {
  const timetable = video.classroomId ? await getTimetable(video.classroomId) : [];
  const schedule = resolveSchedule(video, timetable, tz);
  return periodOffsets(clockInput(video, schedule), tz);
}
