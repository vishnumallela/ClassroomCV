import { useMutation, useQueryClient } from "@tanstack/react-query";
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

export interface Punctuality {
  timezone: string;
  arrivalAt: string | null;
  departureAt: string | null;
  arrivalMinutesLate: number | null;
  departureMinutesLate: number | null;
  presenceShareOfPeriod: number | null;
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

export function LessonDetailsCard({
  videoId,
  lesson,
  punctuality,
}: {
  videoId: string;
  lesson: LessonDetails;
  punctuality: Punctuality;
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
          <Row label="Teacher arrived" value={punctuality.arrivalAt ?? "—"} muted={!anchored} />
          <Row
            label="Against the bell"
            value={hasSchedule ? againstBell(punctuality.arrivalMinutesLate) : "Not Observed"}
            muted={!hasSchedule}
          />
          <Row label="Teacher left" value={punctuality.departureAt ?? "—"} muted={!anchored} />
          <Row
            label="Against the bell"
            value={hasSchedule ? againstBell(punctuality.departureMinutesLate) : "Not Observed"}
            muted={!hasSchedule}
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
