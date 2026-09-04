import type { ResolvedSchedule, TimetableRow } from "@api/lib/timetable";

/**
 * The classroom's day as PERIODS, not files (docs/lesson-coverage-plan.md,
 * Phase D — the register half).
 *
 * A lesson is a period on the classroom's day; a file is coverage of it. So a
 * period's row is assembled from two kinds of evidence:
 *
 *  - its OWN files: recordings the timetable places in this period. Their
 *    attributed teacher's numbers are this period's numbers, and only hers —
 *    the previous period's teacher, however long she stayed, is not in them.
 *  - SPILL-OVER from the NEXT period's file: an adult who was in the room at
 *    that bell and left while the next teacher remained is this period's
 *    teacher finishing. Her departure there is this period's departure, and
 *    how far past her own bell she stayed is this period's over-run.
 *
 * Every number carries a state. "observed" is read from a file; "not
 * observed" says which file is missing. Nothing is guessed from the bell.
 */

export type RegisterState = "observed" | "not_observed";

export interface RegisterVideo {
  id: string;
  title: string;
  status: string;
  schedule: ResolvedSchedule;
  /** The detail DTO's blocks, present when the analysis is done. */
  punctuality: {
    arrivalAt: string | null;
    departureAt: string | null;
    arrivalMinutesLate: number | null;
    departureMinutesLate: number | null;
    presenceShareOfPeriod: number | null;
    notObservedReason: string | null;
  } | null;
  arc: {
    start: { value: number | null; state: string };
    end: { value: number | null; state: string };
    startDelayMin: { value: number | null };
    overrunMin: { value: number | null };
  } | null;
  previousTeacher: {
    state: string;
    departureAt: string | null;
    departureMinutesIntoPeriod: number | null;
    previousPeriodLabel: string | null;
    previousPeriodEnd: string | null;
    departureMinutesAfterHerBell: number | null;
    breakMinutesBeforeThisPeriod: number | null;
    presenceMsInThisFile: number | null;
    boardMsInThisFile: number | null;
  } | null;
}

export interface RegisterRow {
  slot: number;
  label: string;
  scheduledStart: string;
  scheduledEnd: string;
  subject: string | null;
  teacher: string | null;
  /** Files the timetable places in this period. */
  covering: { id: string; title: string; status: string; source: ResolvedSchedule["source"] }[];
  /** This period's teacher, from her own file. */
  own: {
    videoId: string;
    arrivalAt: string | null;
    arrivalMinutesLate: number | null;
    departureAt: string | null;
    departureMinutesLate: number | null;
    presenceShareOfPeriod: number | null;
    lessonStartDelayMin: number | null;
    lessonOverrunMin: number | null;
    notObservedReason: string | null;
  } | null;
  /** This period's teacher finishing inside the NEXT period's file. */
  spillover: {
    fromVideoId: string;
    fromVideoTitle: string;
    departureAt: string | null;
    minutesAfterHerBell: number | null;
    breakMinutes: number | null;
    minutesIntoNextPeriod: number | null;
    presenceMs: number | null;
    boardMs: number | null;
  } | null;
  arrival: { state: RegisterState; reason: string | null };
  departure: { state: RegisterState; reason: string | null; from: "own" | "spillover" | null };
  /** Minutes past this period's end bell she was still in the room (positive),
   *  or minutes before it that she left (negative). Null when not observed. */
  overrunMin: number | null;
}

function normalise(label: string): string {
  return label.trim().toLowerCase().replace(/\s+/g, " ");
}

export function buildDay(periods: TimetableRow[], videos: RegisterVideo[]): RegisterRow[] {
  const rows = [...periods].sort((a, b) => a.slot - b.slot);
  return rows.map((p) => {
    const covering = videos.filter(
      (v) => v.schedule.period && normalise(v.schedule.period) === normalise(p.label),
    );
    const done = covering.find((v) => v.status === "done" && v.punctuality);
    const own: RegisterRow["own"] = done
      ? {
          videoId: done.id,
          arrivalAt: done.punctuality!.arrivalAt,
          arrivalMinutesLate: done.punctuality!.arrivalMinutesLate,
          departureAt: done.punctuality!.departureAt,
          departureMinutesLate: done.punctuality!.departureMinutesLate,
          presenceShareOfPeriod: done.punctuality!.presenceShareOfPeriod,
          lessonStartDelayMin: done.arc?.startDelayMin.value ?? null,
          lessonOverrunMin: done.arc?.overrunMin.value ?? null,
          notObservedReason: done.punctuality!.notObservedReason,
        }
      : null;

    // The next period's file that saw this period's teacher leave.
    const spill = videos.find(
      (v) =>
        v.previousTeacher?.state === "observed" &&
        v.previousTeacher.previousPeriodLabel &&
        normalise(v.previousTeacher.previousPeriodLabel) === normalise(p.label),
    );
    const spillover: RegisterRow["spillover"] = spill?.previousTeacher
      ? {
          fromVideoId: spill.id,
          fromVideoTitle: spill.title,
          departureAt: spill.previousTeacher.departureAt,
          minutesAfterHerBell: spill.previousTeacher.departureMinutesAfterHerBell,
          breakMinutes: spill.previousTeacher.breakMinutesBeforeThisPeriod,
          minutesIntoNextPeriod: spill.previousTeacher.departureMinutesIntoPeriod,
          presenceMs: spill.previousTeacher.presenceMsInThisFile,
          boardMs: spill.previousTeacher.boardMsInThisFile,
        }
      : null;

    const noFile = "No recording covers this period.";
    const arrival: RegisterRow["arrival"] = own
      ? own.arrivalAt
        ? { state: "observed", reason: null }
        : {
            state: "not_observed",
            reason: own.notObservedReason ?? "The recording has no anchor or no teacher timeline.",
          }
      : {
          state: "not_observed",
          reason: covering.length > 0 ? "The recording is not analysed yet." : noFile,
        };

    const departure: RegisterRow["departure"] = own?.departureAt
      ? { state: "observed", reason: null, from: "own" }
      : spillover?.departureAt
        ? {
            state: "observed",
            reason: `Seen leaving in the next period's recording (${spillover.fromVideoTitle}).`,
            from: "spillover",
          }
        : {
            state: "not_observed",
            reason: own ? (own.notObservedReason ?? "No departure in the recording.") : noFile,
            from: null,
          };

    const overrunMin =
      departure.from === "own"
        ? own!.departureMinutesLate
        : departure.from === "spillover"
          ? spillover!.minutesAfterHerBell
          : null;

    return {
      slot: p.slot,
      label: p.label,
      scheduledStart: p.scheduledStart,
      scheduledEnd: p.scheduledEnd,
      subject: p.subject,
      teacher: p.teacher,
      covering: covering.map((v) => ({
        id: v.id,
        title: v.title,
        status: v.status,
        source: v.schedule.source,
      })),
      own,
      spillover,
      arrival,
      departure,
      overrunMin,
    };
  });
}
