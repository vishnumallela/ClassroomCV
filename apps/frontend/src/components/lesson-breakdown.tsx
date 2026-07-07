import type { RouterOutputs } from "@classroom/api-contracts";
import { Card } from "@/components/ui/card";
import { avgStudentsWhileOccupied, lessonStats } from "@/lib/analytics";
import { msToClock } from "@/lib/format";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;
type Interval = [number, number];

const pct = (part: number, whole: number) => (whole > 0 ? (part / whole) * 100 : 0);

const SLICES = [
  { key: "board", label: "At board", color: "bg-amber-400" },
  { key: "circulating", label: "Circulating", color: "bg-primary" },
  { key: "absent", label: "Out of room", color: "bg-zinc-300 dark:bg-zinc-700" },
] as const;

export function LessonBreakdown({
  analytics,
  durationMs,
}: {
  analytics: Analytics;
  durationMs: number | null;
}) {
  if (!durationMs || durationMs <= 0) return null;
  const stats = lessonStats(
    analytics.presenceIntervals as Interval[],
    analytics.boardIntervals as Interval[],
    durationMs,
  );
  const parts = {
    board: stats.boardMs,
    circulating: stats.circulatingMs,
    absent: stats.absentMs,
  };
  const occupied = avgStudentsWhileOccupied(analytics.occupancy);

  const chips: { label: string; value: string }[] = [
    {
      label: "Board share of teaching",
      value: stats.boardShare !== null ? `${Math.round(stats.boardShare * 100)}%` : "n/a",
    },
    { label: "Longest unbroken presence", value: msToClock(stats.longestPresentMs) },
    { label: "Presence segments", value: String(stats.presenceSegments) },
    { label: "Total out of room", value: msToClock(stats.absentMs) },
    {
      label: "Avg students while occupied",
      value: occupied !== null ? occupied.toFixed(1) : "n/a",
    },
  ];

  return (
    <Card className="p-4">
      <div className="text-sm font-medium text-muted-foreground">Lesson breakdown</div>

      <div className="mt-3 flex h-6 w-full overflow-hidden rounded-md">
        {SLICES.map((s) => {
          const p = pct(parts[s.key], durationMs);
          if (p <= 0) return null;
          return (
            <div
              key={s.key}
              className={`flex items-center justify-center ${s.color}`}
              style={{ width: `${p}%` }}
              title={`${s.label}: ${msToClock(parts[s.key])} (${Math.round(p)}%)`}
            >
              {p >= 8 && (
                <span className="px-1 text-[10px] font-medium text-background tabular-nums">
                  {Math.round(p)}%
                </span>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {SLICES.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span className={`size-2 rounded-full ${s.color}`} />
            {s.label} · {msToClock(parts[s.key])}
          </span>
        ))}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {chips.map((c) => (
          <div key={c.label}>
            <div className="text-lg font-semibold tabular-nums">{c.value}</div>
            <div className="mt-0.5 text-xs text-muted-foreground">{c.label}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}
