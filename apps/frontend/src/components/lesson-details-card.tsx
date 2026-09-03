import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CircleAlert } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { orpcClient } from "@/lib/orpc";

const FIELD =
  "w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25";

const ROOM_TYPES = [
  { value: "classroom", label: "Classroom" },
  { value: "lab", label: "Lab" },
  { value: "pe", label: "PE" },
  { value: "library", label: "Library" },
  { value: "practical", label: "Practical" },
] as const;

type RoomType = (typeof ROOM_TYPES)[number]["value"];

function asRoomType(value: string | null): RoomType {
  return ROOM_TYPES.some((r) => r.value === value) ? (value as RoomType) : "classroom";
}

export interface LessonDetails {
  recordingStartedAt: string | null;
  lessonDate: string | null;
  period: string | null;
  scheduledStart: string | null;
  scheduledEnd: string | null;
  subject: string | null;
  yearGroup: string | null;
  roomType: string | null;
  hasFollowingPeriod: boolean | null;
}

export interface PreviousTeacher {
  state: "observed" | "withheld" | "none" | "not_observed";
  reason: string;
  departureAt: string | null;
  departureMinutesIntoPeriod: number | null;
  adultsAtBell: number;
  periodStart: string | null;
  previousPeriodEndKnown: boolean;
}

export interface Punctuality {
  timezone: string;
  arrivalAt: string | null;
  departureAt: string | null;
  arrivalMinutesLate: number | null;
  departureMinutesLate: number | null;
  presenceShareOfPeriod: number | null;
  /**
   * Why these numbers are withheld, when the reason is not visible on the form.
   *
   * A missing scheduled time explains itself — the field next to it is empty.
   * A timeline blended from two adults does not: the analysis looks complete
   * and confident, so without this line the refusal is indistinguishable from
   * "nobody typed the bell times in yet" and gets fixed by typing them in.
   */
  notObservedReason: string | null;
}

/** "6.8 min late" / "6.2 min early" / "on time". Sign carries the meaning. */
function againstBell(minutes: number | null): string {
  if (minutes === null) return "—";
  if (Math.abs(minutes) < 0.5) return "on time";
  const rounded = Math.abs(Math.round(minutes * 10) / 10);
  return `${rounded} min ${minutes > 0 ? "late" : "early"}`;
}

/** Postgres hands back "11:15:00"; an <input type="time"> wants "11:15". */
function toTimeInput(value: string | null): string {
  return value ? value.slice(0, 5) : "";
}

function Row({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className={`text-sm tabular-nums ${muted ? "text-muted-foreground" : "font-medium"}`}>
        {value}
      </span>
    </div>
  );
}

/** "5.6 min into the period" / "3.3 min before it began". */
function intoThePeriod(minutes: number | null): string {
  if (minutes === null) return "—";
  const rounded = Math.abs(Math.round(minutes * 10) / 10);
  if (rounded < 0.5) return "as the period began";
  return minutes > 0 ? `${rounded} min into the period` : `${rounded} min before it began`;
}

export function LessonDetailsCard({
  videoId,
  lesson,
  punctuality,
  previousTeacher,
}: {
  videoId: string;
  lesson: LessonDetails;
  punctuality: Punctuality;
  previousTeacher?: PreviousTeacher;
}) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [form, setForm] = useState({
    period: lesson.period ?? "",
    scheduledStart: toTimeInput(lesson.scheduledStart),
    scheduledEnd: toTimeInput(lesson.scheduledEnd),
    subject: lesson.subject ?? "",
    yearGroup: lesson.yearGroup ?? "",
    roomType: asRoomType(lesson.roomType),
    lessonDate: lesson.lessonDate ?? "",
    hasFollowingPeriod: lesson.hasFollowingPeriod,
  });

  const save = useMutation({
    mutationFn: () =>
      orpcClient.videos.setLessonDetails({
        id: videoId,
        // "" clears the column; the API treats null as an explicit erase, so a
        // blanked field really does empty rather than silently persisting.
        period: form.period.trim() || null,
        scheduledStart: form.scheduledStart || null,
        scheduledEnd: form.scheduledEnd || null,
        subject: form.subject.trim() || null,
        yearGroup: form.yearGroup.trim() || null,
        roomType: form.roomType,
        lessonDate: form.lessonDate || null,
        hasFollowingPeriod: form.hasFollowingPeriod,
      }),
    onSuccess: () => {
      setError(null);
      setOpen(false);
      void queryClient.invalidateQueries();
    },
    onError: (err: unknown) => {
      setError(err instanceof Error ? err.message : "Could not save lesson details.");
    },
  });

  const hasSchedule = Boolean(lesson.scheduledStart && lesson.scheduledEnd);
  const anchored = punctuality.arrivalAt !== null;
  // The measurement was refused rather than missing an input. Every Group A row
  // reads "Not Observed" instead of an em dash, because the dash means "we have
  // nothing to compute from" and this means "we could compute it and it would
  // be wrong".
  const refused = punctuality.notObservedReason !== null;

  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="font-display text-base font-semibold tracking-tight">Lesson details</h2>
          <p className="text-xs text-muted-foreground">
            {lesson.subject || lesson.period ? (
              <>{[lesson.period, lesson.subject, lesson.yearGroup].filter(Boolean).join(" · ")}</>
            ) : (
              "Not entered yet"
            )}
          </p>
        </div>
        <Button size="sm" variant="outline" onClick={() => setOpen((v) => !v)}>
          {open ? "Cancel" : hasSchedule ? "Edit" : "Add details"}
        </Button>
      </div>

      {!open && (
        <div className="mt-4 divide-y divide-border/60">
          <Row
            label="Recording started"
            value={
              punctuality.arrivalAt && lesson.recordingStartedAt
                ? `${new Date(lesson.recordingStartedAt).toLocaleTimeString("en-GB", {
                    timeZone: punctuality.timezone,
                    hour: "2-digit",
                    minute: "2-digit",
                  })} (${punctuality.timezone})`
                : "Unknown — enter it to measure punctuality"
            }
            muted={!lesson.recordingStartedAt}
          />
          <Row
            label="Scheduled"
            value={
              hasSchedule
                ? `${toTimeInput(lesson.scheduledStart)} – ${toTimeInput(lesson.scheduledEnd)}`
                : "Not entered"
            }
            muted={!hasSchedule}
          />
          <Row
            label="Teacher arrived"
            value={refused ? "Not Observed" : (punctuality.arrivalAt ?? "—")}
            muted={!anchored}
          />
          <Row
            label="Against the bell"
            value={
              hasSchedule && !refused ? againstBell(punctuality.arrivalMinutesLate) : "Not Observed"
            }
            muted={!hasSchedule || refused}
          />
          <Row
            label="Teacher left"
            value={refused ? "Not Observed" : (punctuality.departureAt ?? "—")}
            muted={!anchored}
          />
          <Row
            label="Against the bell"
            value={
              hasSchedule && !refused
                ? againstBell(punctuality.departureMinutesLate)
                : "Not Observed"
            }
            muted={!hasSchedule || refused}
          />
          <Row
            label="Present, of the period"
            value={
              punctuality.presenceShareOfPeriod !== null
                ? `${Math.round(punctuality.presenceShareOfPeriod * 100)}%`
                : "Not Observed"
            }
            muted={punctuality.presenceShareOfPeriod === null}
          />
        </div>
      )}

      {/* The other lesson in this file. A recording that starts on the bell with
          the previous teacher still in the room shows the END of her lesson —
          when she left, and whether she stayed to her own bell. Shown only when
          there is something to say; a single-teacher lesson renders nothing. */}
      {!open && previousTeacher && previousTeacher.state === "observed" && (
        <div className="mt-4 border-t border-border/60 pt-3">
          <p className="mb-1 text-xs font-medium text-muted-foreground">
            Previous period&rsquo;s teacher
          </p>
          <div className="divide-y divide-border/60">
            <Row label="Left the room" value={previousTeacher.departureAt ?? "—"} />
            <Row
              label={`Against this period's ${previousTeacher.periodStart ?? ""} bell`}
              value={intoThePeriod(previousTeacher.departureMinutesIntoPeriod)}
            />
          </div>
          <p className="mt-2 text-[0.7rem] leading-relaxed text-muted-foreground">
            {previousTeacher.reason} Only the end of her stay is in this recording. Whether she
            stayed to her own bell needs her period&rsquo;s times (the timetable) and the previous
            file.
          </p>
        </div>
      )}
      {!open && previousTeacher && previousTeacher.state === "withheld" && (
        <p className="mt-3 flex items-start gap-1.5 rounded-lg bg-muted/50 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
          <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-tier-medium" />
          <span>Previous period&rsquo;s teacher: {previousTeacher.reason}</span>
        </p>
      )}

      {!open && punctuality.notObservedReason && (
        <p className="mt-3 flex items-start gap-1.5 rounded-lg bg-muted/50 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
          <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-tier-medium" />
          <span>{punctuality.notObservedReason}</span>
        </p>
      )}

      {open && (
        <form
          className="mt-4 space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            save.mutate();
          }}
        >
          <div className="grid grid-cols-2 gap-3">
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Period</span>
              <input
                className={FIELD}
                placeholder="Period 3"
                value={form.period}
                onChange={(e) => setForm((f) => ({ ...f, period: e.target.value }))}
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Lesson date</span>
              <input
                type="date"
                className={FIELD}
                value={form.lessonDate}
                onChange={(e) => setForm((f) => ({ ...f, lessonDate: e.target.value }))}
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Scheduled start</span>
              <input
                type="time"
                className={FIELD}
                value={form.scheduledStart}
                onChange={(e) => setForm((f) => ({ ...f, scheduledStart: e.target.value }))}
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Scheduled end</span>
              <input
                type="time"
                className={FIELD}
                value={form.scheduledEnd}
                onChange={(e) => setForm((f) => ({ ...f, scheduledEnd: e.target.value }))}
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">
                Subject <span className="font-normal text-muted-foreground">(optional)</span>
              </span>
              <input
                className={FIELD}
                placeholder="Biology"
                value={form.subject}
                onChange={(e) => setForm((f) => ({ ...f, subject: e.target.value }))}
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">
                Year group <span className="font-normal text-muted-foreground">(optional)</span>
              </span>
              <input
                className={FIELD}
                placeholder="Class 12"
                value={form.yearGroup}
                onChange={(e) => setForm((f) => ({ ...f, yearGroup: e.target.value }))}
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Room type</span>
              <select
                className={FIELD}
                value={form.roomType}
                onChange={(e) => setForm((f) => ({ ...f, roomType: asRoomType(e.target.value) }))}
              >
                {ROOM_TYPES.map((r) => (
                  <option key={r.value} value={r.value}>
                    {r.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2.5 self-end rounded-lg border border-border px-3 py-2.5">
              <input
                type="checkbox"
                className="size-4"
                checked={form.hasFollowingPeriod === true}
                onChange={(e) =>
                  setForm((f) => ({ ...f, hasFollowingPeriod: e.target.checked ? true : null }))
                }
              />
              <span className="text-sm">Class continues next period</span>
            </label>
          </div>

          <p className="text-xs text-muted-foreground">
            Scheduled times are local to {punctuality.timezone}. Leave them blank and punctuality
            reads “Not Observed” rather than guessing.
          </p>

          {error && <p className="text-sm text-destructive">{error}</p>}

          <div className="flex justify-end gap-2 pt-1">
            <Button type="button" size="sm" variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button type="submit" size="sm" disabled={save.isPending}>
              {save.isPending ? "Saving…" : "Save"}
            </Button>
          </div>
        </form>
      )}
    </Card>
  );
}
