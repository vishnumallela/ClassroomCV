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
};

export function EventsTable({
  events,
  onSeek,
}: {
  events: VideoEvent[];
  onSeek: (ms: number) => void;
}) {
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
              <TableCell className="font-medium">{KIND_LABEL[e.kind] ?? e.kind}</TableCell>
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
