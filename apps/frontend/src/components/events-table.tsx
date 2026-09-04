import type { RouterOutputs } from "@classroom/api-contracts";
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

type VideoEvent = RouterOutputs["videos"]["get"]["events"][number];

// Exported so the timeline ticks describe each moment with the same wording
// as this table; the two views must never drift apart.
export const KIND_LABEL: Record<string, string> = {
  enter: "Entered room",
  exit: "Left room",
  board_enter: "Arrived at board",
  board_leave: "Left board",
  pointing_start: "Started pointing",
  pointing_end: "Stopped pointing",
  writing_start: "Started writing",
  writing_end: "Stopped writing",
};

const METHOD_LABEL: Record<string, string> = {
  door: "seen at the door",
  buffer: "inferred: out of frame beyond the buffer",
  start: "first sighting",
};

export function EventsTable({
  events,
  entryExit = [],
  onSeek,
}: {
  events: VideoEvent[];
  /** How each enter/exit was decided (analytics.entryExit), matched by time. */
  entryExit?: { kind: string; ts_ms: number; method?: string | null }[];
  onSeek: (ms: number) => void;
}) {
  const methodAt = (kind: string, ms: number): string | null => {
    const hit = entryExit.find((e) => e.kind === kind && e.ts_ms === ms);
    return hit?.method ? (METHOD_LABEL[hit.method] ?? hit.method) : null;
  };
  if (events.length === 0) {
    return <Card className="p-6 text-sm text-muted-foreground">No teacher events detected.</Card>;
  }
  return (
    <Card>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Event</TableHead>
            <TableHead>Track</TableHead>
            <TableHead>Time</TableHead>
            <TableHead className="text-right">
              <span className="sr-only">Actions</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {events.map((e) => (
            <TableRow key={`${e.kind}-${e.videoTsMs}-${e.trackNo}`}>
              <TableCell className="font-medium">
                {KIND_LABEL[e.kind] ?? e.kind}
                {methodAt(e.kind, e.videoTsMs) && (
                  <span className="ml-1.5 text-xs text-muted-foreground">
                    ({methodAt(e.kind, e.videoTsMs)})
                  </span>
                )}
              </TableCell>
              <TableCell className="tabular-nums text-muted-foreground">
                {e.trackNo !== null ? `#${e.trackNo}` : "n/a"}
              </TableCell>
              <TableCell className="tabular-nums">{msToClock(e.videoTsMs)}</TableCell>
              <TableCell className="text-right">
                <Button size="sm" variant="ghost" onClick={() => onSeek(e.videoTsMs)}>
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
