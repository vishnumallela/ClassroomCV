import type { RouterOutputs } from "@classroom/api-contracts";
import { Card } from "@/components/ui/card";
import { msToClock, percentOf } from "@/lib/format";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;

export function KpiCards({
  analytics,
  durationMs,
}: {
  analytics: Analytics;
  durationMs: number | null;
}) {
  const hasBoard = analytics.teacherBoardMs !== null;
  const tiles = [
    {
      label: "Teacher in class",
      value: msToClock(analytics.teacherPresentMs),
      sub: percentOf(analytics.teacherPresentMs, durationMs) + " of lesson",
    },
    {
      label: "Time at board",
      value: hasBoard ? msToClock(analytics.teacherBoardMs) : "n/a",
      sub: hasBoard
        ? percentOf(analytics.teacherBoardMs, durationMs) + " of lesson"
        : "no board zone",
    },
    { label: "Entries", value: String(analytics.entries), sub: "into the room" },
    { label: "Exits", value: String(analytics.exits), sub: "out of the room" },
    {
      label: "Avg students",
      value: analytics.avgStudents !== null ? analytics.avgStudents.toFixed(1) : "n/a",
      sub: "present per second",
    },
    {
      label: "Peak students",
      value: analytics.maxStudents !== null ? String(analytics.maxStudents) : "n/a",
      sub: "at once",
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      {tiles.map((t) => (
        <Card key={t.label} className="p-4">
          <div className="text-xs text-muted-foreground">{t.label}</div>
          <div className="mt-1 text-xl font-semibold tabular-nums">{t.value}</div>
          <div className="mt-0.5 text-xs text-muted-foreground">{t.sub}</div>
        </Card>
      ))}
    </div>
  );
}
