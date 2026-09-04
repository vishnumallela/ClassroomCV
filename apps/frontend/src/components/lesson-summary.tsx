import type { RouterOutputs } from "@classroom/api-contracts";
import { Card } from "@/components/ui/card";
import { Stat, type MeasureState } from "@/components/ui/stat";
import { againstBell, clockAt, percent } from "@/lib/measures";

type Detail = RouterOutputs["videos"]["get"];

/**
 * The lesson at a glance: the five answers the school asks first, in the
 * order they happen. Everything below the strip is the evidence for these.
 */
export function LessonSummary({ data }: { data: Detail }) {
  const { punctuality: p, arc, lesson } = data;
  const tz = p.timezone;
  const anchor = lesson.recordingStartedAt;
  const schedule = lesson.schedule;
  const refused = p.notObservedReason !== null;
  const bells = schedule.scheduledStart && schedule.scheduledEnd;

  const attendance = (value: string | null, minutes: number | null): [string, MeasureState] =>
    refused || value === null
      ? ["", "not_observed"]
      : [bells ? againstBell(minutes) : "no bell to compare", "observed"];
  const [arrivedSub, arrivedState] = attendance(p.arrivalAt, p.arrivalMinutesLate);
  const [leftSub, leftState] = attendance(p.departureAt, p.departureMinutesLate);

  const startClock = clockAt(anchor, arc.start.value, tz);
  const endClock = clockAt(anchor, arc.end.value, tz);
  const startState = arc.start.state as MeasureState;
  const endState = arc.end.state as MeasureState;

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="font-display text-base font-semibold tracking-tight">At a glance</h2>
        <p className="text-xs text-muted-foreground">
          {[schedule.period, schedule.subject, schedule.yearGroup].filter(Boolean).join(" · ") ||
            "Lesson details not entered"}
          {bells
            ? ` · bells ${schedule.scheduledStart!.slice(0, 5)}–${schedule.scheduledEnd!.slice(0, 5)}` +
              (schedule.source === "timetable" ? " (timetable)" : "")
            : " · no bells"}
          {lesson.lessonDate ? ` · ${lesson.lessonDate}` : ""}
        </p>
      </div>
      <div className="mt-4 grid gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <Stat
          id="R1 R2"
          label="Teacher arrived"
          value={p.arrivalAt ?? "—"}
          sub={arrivedSub}
          state={arrivedState}
          reason={p.notObservedReason}
          size="lg"
        />
        <Stat
          id="R7 R8"
          label="Lesson started"
          value={startClock ?? "—"}
          sub={
            arc.startDelayMin.value !== null
              ? againstBell(arc.startDelayMin.value, "after the bell", "before the bell")
              : arc.start.value !== null
                ? "no bell to compare"
                : ""
          }
          state={startState}
          reason={arc.start.reason}
          size="lg"
        />
        <Stat
          id="R9 R12"
          label="Lesson ended"
          value={endClock ?? "—"}
          sub={
            arc.overrunMin.value !== null
              ? againstBell(arc.overrunMin.value, "past the bell", "before the bell", "on the bell")
              : arc.end.value !== null
                ? "no bell to compare"
                : ""
          }
          state={endState}
          reason={arc.end.reason}
          size="lg"
        />
        <Stat
          id="R3 R4"
          label="Teacher left"
          value={p.departureAt ?? "—"}
          sub={leftSub}
          state={leftState}
          reason={p.notObservedReason}
          size="lg"
        />
        <Stat
          id="R5"
          label="In the room"
          value={percent(p.presenceShareOfPeriod)}
          sub={p.presenceShareOfPeriod !== null ? "of the scheduled period" : ""}
          state={p.presenceShareOfPeriod !== null ? "observed" : "not_observed"}
          reason={
            p.notObservedReason ?? (bells ? null : "The bells are not known for this lesson.")
          }
          size="lg"
        />
      </div>
    </Card>
  );
}
