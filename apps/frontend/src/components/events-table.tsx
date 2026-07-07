import type { RouterOutputs } from "@classroom/api-contracts";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { msToClock } from "@/lib/format";

type VideoEvent = RouterOutputs["videos"]["get"]["events"][number];

const KIND_LABEL: Record<string, string> = {
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
    <Card className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-border text-left text-xs text-muted-foreground">
            <tr>
              <th className="px-4 py-2 font-medium">Event</th>
              <th className="px-4 py-2 font-medium">Track</th>
              <th className="px-4 py-2 font-medium">Time</th>
              <th className="px-4 py-2">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {events.map((e) => (
              <tr key={`${e.kind}-${e.videoTsMs}-${e.trackNo}`} className="hover:bg-accent/40">
                <td className="px-4 py-2">{KIND_LABEL[e.kind] ?? e.kind}</td>
                <td className="px-4 py-2 tabular-nums text-muted-foreground">
                  {e.trackNo !== null ? `#${e.trackNo}` : "n/a"}
                </td>
                <td className="px-4 py-2 tabular-nums">{msToClock(e.videoTsMs)}</td>
                <td className="px-4 py-2 text-right">
                  <Button size="sm" variant="ghost" onClick={() => onSeek(e.videoTsMs)}>
                    Jump
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
