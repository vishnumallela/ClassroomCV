import type { RouterOutputs } from "@classroom/api-contracts";
import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { CalendarCheck } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { msToClock } from "@/lib/format";
import { orpc } from "@/lib/orpc";

export const Route = createFileRoute("/classrooms/$id/register")({ component: ClassroomRegister });

type Row = RouterOutputs["lessons"]["day"]["rows"][number];

function hhmm(t: string): string {
  return t.slice(0, 5);
}

function minutes(v: number | null, late: string, early: string): string {
  if (v === null) return "—";
  if (Math.abs(v) < 0.5) return "on the bell";
  return `${Math.abs(Math.round(v * 10) / 10)} min ${v > 0 ? late : early}`;
}

function NotObserved({ reason }: { reason: string | null }) {
  return (
    <span className="text-muted-foreground" title={reason ?? undefined}>
      Not Observed
    </span>
  );
}

/**
 * The register: the classroom's day as periods. A period's numbers come from
 * its own file, and only from its own teacher; its departure and over-run can
 * also come from the NEXT period's file, where she was seen finishing. That
 * is how the period-2 teacher's last minutes, recorded in the period-3 file,
 * are credited to period 2 and kept out of period 3.
 */
function ClassroomRegister() {
  const { id } = Route.useParams();
  const [date, setDate] = useState<string | undefined>(undefined);
  const { data, isLoading, isError } = useQuery(
    orpc.lessons.day.queryOptions({ input: { classroomId: id, date } }),
  );

  if (isLoading) return <Skeleton className="h-64 rounded-xl" />;
  if (isError || !data)
    return <Card className="p-6 text-sm text-destructive">Could not load the register.</Card>;

  return (
    <div className="space-y-4">
      <Card className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div className="space-y-1">
          <h2 className="flex items-center gap-2 font-display text-base font-semibold tracking-tight">
            <CalendarCheck className="size-4 text-muted-foreground" />
            Register
          </h2>
          <p className="text-xs text-muted-foreground">
            One row per period of the day. A period's numbers are its own teacher's; a departure
            seen in the next period's recording is credited back to the period it belongs to.
          </p>
        </div>
        {data.dates.length > 0 && (
          <label className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Day</span>
            <select
              value={data.date ?? ""}
              onChange={(e) => setDate(e.target.value)}
              className="rounded-md border border-input bg-background px-2 py-1 text-sm"
            >
              {data.dates.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
        )}
      </Card>

      {data.rows.length === 0 ? (
        <Card className="p-6 text-sm text-muted-foreground">
          {data.date
            ? "No timetable rows for this weekday. Add the classroom's timetable under Configuration."
            : "No dated lessons yet. Upload a recording, or type its date on the lesson page."}
        </Card>
      ) : (
        <Card className="overflow-x-auto p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Period</TableHead>
                <TableHead>Bells</TableHead>
                <TableHead>Teacher</TableHead>
                <TableHead>Recording</TableHead>
                <TableHead>Arrived</TableHead>
                <TableHead>Left</TableHead>
                <TableHead>Over / under-run</TableHead>
                <TableHead>In the room</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.rows.map((r: Row) => (
                <TableRow key={r.slot}>
                  <TableCell className="whitespace-nowrap">
                    <span className="font-medium">{r.label}</span>
                    {r.subject && (
                      <span className="block text-xs text-muted-foreground">{r.subject}</span>
                    )}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-muted-foreground">
                    {hhmm(r.scheduledStart)} – {hhmm(r.scheduledEnd)}
                  </TableCell>
                  <TableCell>
                    {r.teacher ?? <span className="text-muted-foreground">—</span>}
                  </TableCell>
                  <TableCell>
                    {r.covering.length === 0 ? (
                      <span className="text-muted-foreground">none</span>
                    ) : (
                      r.covering.map((c) => (
                        <Link
                          key={c.id}
                          to="/videos/$id"
                          params={{ id: c.id }}
                          className="block truncate underline-offset-2 hover:underline"
                        >
                          {c.title}
                          {c.status !== "done" && (
                            <Badge variant="secondary" className="ml-1 text-[0.6rem]">
                              {c.status}
                            </Badge>
                          )}
                        </Link>
                      ))
                    )}
                    {r.spillover && (
                      <Link
                        to="/videos/$id"
                        params={{ id: r.spillover.fromVideoId }}
                        className="block truncate text-xs text-muted-foreground underline-offset-2 hover:underline"
                      >
                        + seen leaving in {r.spillover.fromVideoTitle}
                      </Link>
                    )}
                  </TableCell>
                  <TableCell className="whitespace-nowrap">
                    {r.arrival.state === "observed" && r.own ? (
                      <>
                        <span className="font-medium">{r.own.arrivalAt}</span>
                        <span className="block text-xs text-muted-foreground">
                          {minutes(r.own.arrivalMinutesLate, "late", "early")}
                        </span>
                      </>
                    ) : (
                      <NotObserved reason={r.arrival.reason} />
                    )}
                  </TableCell>
                  <TableCell className="whitespace-nowrap">
                    {r.departure.state === "observed" ? (
                      <>
                        <span className="font-medium">
                          {r.departure.from === "own"
                            ? r.own?.departureAt
                            : r.spillover?.departureAt}
                        </span>
                        <span className="block text-xs text-muted-foreground">
                          {r.departure.from === "spillover"
                            ? "from the next recording"
                            : minutes(r.own?.departureMinutesLate ?? null, "late", "early")}
                        </span>
                      </>
                    ) : (
                      <NotObserved reason={r.departure.reason} />
                    )}
                  </TableCell>
                  <TableCell className="whitespace-nowrap">
                    {r.overrunMin === null ? (
                      <NotObserved reason={r.departure.reason} />
                    ) : (
                      <>
                        <span
                          className={`font-medium ${r.overrunMin > 0.5 ? "text-tier-medium" : ""}`}
                        >
                          {minutes(r.overrunMin, "over-run", "under-run")}
                        </span>
                        {r.spillover && r.spillover.breakMinutes ? (
                          <span className="block text-xs text-muted-foreground">
                            {r.spillover.breakMinutes} min of it the break; left{" "}
                            {r.spillover.minutesIntoNextPeriod} min into the next period
                          </span>
                        ) : null}
                      </>
                    )}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs">
                    {r.own?.presenceShareOfPeriod !== null &&
                    r.own?.presenceShareOfPeriod !== undefined ? (
                      <span>{Math.round(r.own.presenceShareOfPeriod * 100)}% of the period</span>
                    ) : r.spillover?.presenceMs ? (
                      <span>
                        {msToClock(r.spillover.presenceMs)} after her bell
                        {r.spillover.boardMs
                          ? `, ${msToClock(r.spillover.boardMs)} at the board`
                          : ""}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      {data.undated > 0 && (
        <p className="text-xs text-muted-foreground">
          {data.undated} recording{data.undated === 1 ? " has" : "s have"} no date and no clock
          anchor, so {data.undated === 1 ? "it is" : "they are"} not on any day.
        </p>
      )}
    </div>
  );
}
