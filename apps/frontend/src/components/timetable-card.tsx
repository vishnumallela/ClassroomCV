import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CalendarDays, Copy, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { orpcClient } from "@/lib/orpc";

/**
 * The classroom's week, typed once.
 *
 * Every lesson uploaded to this room resolves its bell times from here unless
 * someone typed bells on the lesson itself, so a correction on this page
 * reaches every lesson in the room with no re-analysis. Breaks are simply the
 * gaps between periods: the API derives "does a period follow this one" and
 * "when did the previous period end" from the rows, which is what lets the
 * previous period's teacher be measured against her own bell.
 */

export interface TimetableRowInput {
  weekday: number;
  slot: number;
  label: string;
  scheduledStart: string;
  scheduledEnd: string;
  subject: string | null;
  teacher: string | null;
  yearGroup: string | null;
}

const WEEKDAYS: { value: number; label: string }[] = [
  { value: 1, label: "Mon" },
  { value: 2, label: "Tue" },
  { value: 3, label: "Wed" },
  { value: 4, label: "Thu" },
  { value: 5, label: "Fri" },
  { value: 6, label: "Sat" },
  { value: 7, label: "Sun" },
];

const FIELD =
  "w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25";

/** "09:50:00" from the API → "09:50" for a time input. */
function toTimeInput(value: string): string {
  return value.length >= 5 ? value.slice(0, 5) : value;
}

function nextLabel(rows: TimetableRowInput[]): string {
  const numbers = rows
    .map((r) => /\d+/.exec(r.label)?.[0])
    .filter((n): n is string => n !== undefined)
    .map(Number);
  const next = numbers.length > 0 ? Math.max(...numbers) + 1 : 1;
  return `Period ${next}`;
}

export function TimetableCard({
  classroomId,
  initialRows,
}: {
  classroomId: string;
  initialRows: TimetableRowInput[];
}) {
  const queryClient = useQueryClient();
  const [rows, setRows] = useState<TimetableRowInput[]>(() =>
    initialRows.map((r) => ({
      ...r,
      scheduledStart: toTimeInput(r.scheduledStart),
      scheduledEnd: toTimeInput(r.scheduledEnd),
    })),
  );
  const [weekday, setWeekday] = useState(1);
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dayRows = rows.filter((r) => r.weekday === weekday).sort((a, b) => a.slot - b.slot);
  const countFor = (d: number) => rows.filter((r) => r.weekday === d).length;

  const edit = (mutate: (rows: TimetableRowInput[]) => TimetableRowInput[]) => {
    setRows((prev) => mutate(prev));
    setDirty(true);
    setError(null);
  };

  const patchRow = (slot: number, patch: Partial<TimetableRowInput>) =>
    edit((prev) =>
      prev.map((r) => (r.weekday === weekday && r.slot === slot ? { ...r, ...patch } : r)),
    );

  const addRow = () =>
    edit((prev) => {
      const last = dayRows.at(-1);
      return [
        ...prev,
        {
          weekday,
          slot: (last?.slot ?? -1) + 1,
          label: nextLabel(dayRows),
          scheduledStart: last?.scheduledEnd ?? "",
          scheduledEnd: "",
          subject: null,
          teacher: null,
          yearGroup: last?.yearGroup ?? null,
        },
      ];
    });

  const removeRow = (slot: number) =>
    edit((prev) => prev.filter((r) => !(r.weekday === weekday && r.slot === slot)));

  // Bells are the same every day at most schools; subjects are not. Copying
  // the day's periods gives the other days their bells in one click, and the
  // subjects can then be corrected per day.
  const copyToOtherDays = () =>
    edit((prev) => {
      const source = prev.filter((r) => r.weekday === weekday);
      const others = WEEKDAYS.filter((d) => d.value !== weekday && d.value !== 7).map(
        (d) => d.value,
      );
      return [
        ...prev.filter((r) => r.weekday === weekday || r.weekday === 7),
        ...others.flatMap((d) => source.map((r) => ({ ...r, weekday: d }))),
      ];
    });

  const invalid = rows.find((r) => !r.label.trim() || !r.scheduledStart || !r.scheduledEnd);
  const backwards = rows.find(
    (r) => r.scheduledStart && r.scheduledEnd && r.scheduledEnd <= r.scheduledStart,
  );

  const save = useMutation({
    mutationFn: () =>
      orpcClient.classrooms.setTimetable({
        id: classroomId,
        rows: rows.map((r) => ({
          weekday: r.weekday,
          slot: r.slot,
          label: r.label.trim(),
          scheduledStart: r.scheduledStart,
          scheduledEnd: r.scheduledEnd,
          subject: r.subject?.trim() || null,
          teacher: r.teacher?.trim() || null,
          yearGroup: r.yearGroup?.trim() || null,
        })),
      }),
    onSuccess: () => {
      setDirty(false);
      setError(null);
      void queryClient.invalidateQueries();
    },
    onError: (err: unknown) => {
      setError(err instanceof Error ? err.message : "Could not save the timetable.");
    },
  });

  return (
    <Card className="space-y-4 p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <h2 className="font-display text-lg font-medium">Timetable</h2>
          <p className="max-w-lg text-sm text-muted-foreground">
            This room's week, typed once. Every lesson uploaded here takes its bell times from it
            unless bells were typed on the lesson itself, and the previous period's teacher is
            measured against her own bell. Breaks are the gaps between periods.
          </p>
        </div>
        <CalendarDays className="size-5 text-muted-foreground" />
      </div>

      <div className="flex flex-wrap gap-1.5">
        {WEEKDAYS.map((d) => (
          <button
            key={d.value}
            type="button"
            onClick={() => setWeekday(d.value)}
            className={
              "rounded-md border px-2.5 py-1 text-xs transition-colors " +
              (d.value === weekday
                ? "border-primary bg-primary text-primary-foreground"
                : "border-input bg-background text-muted-foreground hover:text-foreground")
            }
          >
            {d.label}
            {countFor(d.value) > 0 && <span className="ml-1 opacity-70">{countFor(d.value)}</span>}
          </button>
        ))}
      </div>

      {dayRows.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No periods on this day yet. Add one, or copy another day's periods here.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="text-left text-xs text-muted-foreground">
              <tr>
                <th className="pb-2 pr-2 font-medium">Period</th>
                <th className="pb-2 pr-2 font-medium">Starts</th>
                <th className="pb-2 pr-2 font-medium">Ends</th>
                <th className="pb-2 pr-2 font-medium">Subject</th>
                <th className="pb-2 pr-2 font-medium">Teacher</th>
                <th className="pb-2 pr-2 font-medium">Class</th>
                <th className="pb-2" />
              </tr>
            </thead>
            <tbody>
              {dayRows.map((r) => (
                <tr key={r.slot}>
                  <td className="py-1 pr-2">
                    <input
                      value={r.label}
                      maxLength={60}
                      onChange={(e) => patchRow(r.slot, { label: e.target.value })}
                      className={FIELD}
                    />
                  </td>
                  <td className="py-1 pr-2">
                    <input
                      type="time"
                      value={r.scheduledStart}
                      onChange={(e) => patchRow(r.slot, { scheduledStart: e.target.value })}
                      className={FIELD}
                    />
                  </td>
                  <td className="py-1 pr-2">
                    <input
                      type="time"
                      value={r.scheduledEnd}
                      onChange={(e) => patchRow(r.slot, { scheduledEnd: e.target.value })}
                      className={FIELD}
                    />
                  </td>
                  <td className="py-1 pr-2">
                    <input
                      value={r.subject ?? ""}
                      maxLength={120}
                      placeholder="english"
                      onChange={(e) => patchRow(r.slot, { subject: e.target.value })}
                      className={FIELD}
                    />
                  </td>
                  <td className="py-1 pr-2">
                    <input
                      value={r.teacher ?? ""}
                      maxLength={120}
                      placeholder="from the sheet"
                      onChange={(e) => patchRow(r.slot, { teacher: e.target.value })}
                      className={FIELD}
                    />
                  </td>
                  <td className="py-1 pr-2">
                    <input
                      value={r.yearGroup ?? ""}
                      maxLength={60}
                      placeholder="class 7"
                      onChange={(e) => patchRow(r.slot, { yearGroup: e.target.value })}
                      className={FIELD}
                    />
                  </td>
                  <td className="py-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      aria-label={`Remove ${r.label}`}
                      onClick={() => removeRow(r.slot)}
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={addRow}>
          <Plus className="size-3.5" />
          Add period
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={dayRows.length === 0}
          onClick={copyToOtherDays}
        >
          <Copy className="size-3.5" />
          Copy this day to Mon–Sat
        </Button>
        <Button
          size="sm"
          disabled={!dirty || Boolean(invalid) || Boolean(backwards) || save.isPending}
          onClick={() => save.mutate()}
        >
          {save.isPending ? "Saving…" : "Save timetable"}
        </Button>
        {save.isSuccess && !dirty && <span className="text-xs text-muted-foreground">Saved.</span>}
        {invalid && dirty && (
          <span className="text-xs text-muted-foreground">
            Every period needs a name, a start and an end.
          </span>
        )}
        {backwards && (
          <span className="text-xs text-destructive">
            {backwards.label || "A period"} ends before it starts.
          </span>
        )}
        {error && <span className="text-xs text-destructive">{error}</span>}
      </div>
    </Card>
  );
}
