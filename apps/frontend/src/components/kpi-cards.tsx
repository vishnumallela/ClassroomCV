import type { RouterOutputs } from "@classroom/api-contracts";
import { Card } from "@/components/ui/card";
import { msToClock, percentOf } from "@/lib/format";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;

type Confidence = { label: string; tone: "high" | "medium" | "low" };

function confidenceBadge(conf: number | null | undefined): Confidence | null {
  if (conf === null || conf === undefined) return null;
  const pct = Math.round(conf * 100);
  if (conf >= 0.75) return { label: `${pct}% sure`, tone: "high" };
  if (conf >= 0.6) return { label: `${pct}% sure`, tone: "medium" };
  return { label: `${pct}% sure`, tone: "low" };
}

const TONE_CLASS: Record<Confidence["tone"], string> = {
  high: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  medium: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  low: "bg-red-500/15 text-red-600 dark:text-red-400",
};

export function KpiCards({
  analytics,
  durationMs,
  teacherConfidence,
}: {
  analytics: Analytics;
  durationMs: number | null;
  teacherConfidence?: number | null;
}) {
  const hasBoard = analytics.teacherBoardMs !== null;
  const badge = confidenceBadge(teacherConfidence);
  const tiles = [
    {
      label: "Teacher in class",
      value: msToClock(analytics.teacherPresentMs),
      sub: percentOf(analytics.teacherPresentMs, durationMs) + " of lesson",
      badge,
    },
    {
      label: "Time at board",
      value: hasBoard ? msToClock(analytics.teacherBoardMs) : "n/a",
      sub: hasBoard
        ? percentOf(analytics.teacherBoardMs, durationMs) + " of lesson"
        : "no board zone",
      badge: null,
    },
    { label: "Entries", value: String(analytics.entries), sub: "into the room", badge: null },
    { label: "Exits", value: String(analytics.exits), sub: "out of the room", badge: null },
    {
      label: "Avg students",
      value: analytics.avgStudents !== null ? analytics.avgStudents.toFixed(1) : "n/a",
      sub: "avg concurrent (per 5s)",
      badge: null,
    },
    {
      label: "Peak students",
      value: analytics.maxStudents !== null ? String(analytics.maxStudents) : "n/a",
      sub: "at once",
      badge: null,
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      {tiles.map((t) => (
        <Card key={t.label} className="p-4">
          <div className="flex items-start justify-between gap-1">
            <div className="text-xs text-muted-foreground">{t.label}</div>
            {t.badge && (
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${TONE_CLASS[t.badge.tone]}`}
                title="How confident the classifier is that this identity is the teacher"
              >
                {t.badge.label}
              </span>
            )}
          </div>
          <div className="mt-1 text-xl font-semibold tabular-nums">{t.value}</div>
          <div className="mt-0.5 text-xs text-muted-foreground">{t.sub}</div>
        </Card>
      ))}
    </div>
  );
}
