import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { msToClock } from "@/lib/format";

type Interval = [number, number];

export function BoardSessions({
  boardIntervals,
  onSeek,
}: {
  boardIntervals: Interval[];
  onSeek: (ms: number) => void;
}) {
  const sessions = boardIntervals.toSorted((a, b) => a[0] - b[0]);
  if (sessions.length === 0) {
    return (
      <Card className="p-6 text-sm text-muted-foreground">
        No board sessions detected (no board zone, or the teacher never worked at the board).
      </Card>
    );
  }
  const durations = sessions.map(([s, e]) => e - s);
  const total = durations.reduce((a, b) => a + b, 0);
  const summary = [
    { label: "Sessions", value: String(sessions.length) },
    { label: "Longest", value: msToClock(Math.max(...durations)) },
    { label: "Average", value: msToClock(Math.round(total / sessions.length)) },
    { label: "First at", value: msToClock(sessions[0]![0]) },
  ];

  return (
    <Card>
      <div className="flex flex-wrap gap-x-6 gap-y-2 border-b border-border p-4">
        <div className="text-sm font-medium text-muted-foreground">Board sessions</div>
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          {summary.map((s) => (
            <div key={s.label} className="text-xs text-muted-foreground">
              {s.label}{" "}
              <span className="font-semibold tabular-nums text-foreground">{s.value}</span>
            </div>
          ))}
        </div>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-12">#</TableHead>
            <TableHead>Start</TableHead>
            <TableHead>Duration</TableHead>
            <TableHead className="text-right">
              <span className="sr-only">Actions</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sessions.map(([start, end], i) => (
            <TableRow key={start}>
              <TableCell className="tabular-nums text-muted-foreground">{i + 1}</TableCell>
              <TableCell className="tabular-nums">{msToClock(start)}</TableCell>
              <TableCell className="tabular-nums">{msToClock(end - start)}</TableCell>
              <TableCell className="text-right">
                <Button size="sm" variant="ghost" onClick={() => onSeek(start)}>
                  Jump
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  );
}
