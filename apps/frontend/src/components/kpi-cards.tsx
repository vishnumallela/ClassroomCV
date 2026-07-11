import type { RouterOutputs } from "@classroom/api-contracts";
import type { CSSProperties } from "react";
import { Badge } from "@/components/ui/badge";
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
    {
      label: "Pointing at board",
      value:
        hasBoard && analytics.teacherPointingMs !== null
          ? msToClock(analytics.teacherPointingMs)
          : "n/a",
      sub:
        hasBoard && analytics.teacherPointingMs !== null
          ? percentOf(analytics.teacherPointingMs, durationMs) + " of lesson"
          : "no board zone",
      badge: null,
    },
    {
      label: "Writing on board",
      value:
        hasBoard && analytics.teacherWritingMs !== null
          ? msToClock(analytics.teacherWritingMs)
          : "n/a",
      sub:
        hasBoard && analytics.teacherWritingMs !== null
          ? percentOf(analytics.teacherWritingMs, durationMs) + " of lesson"
          : "no board zone",
      badge: null,
    },
    { label: "Entries", value: String(analytics.entries), sub: "into the room", badge: null },
    { label: "Exits", value: String(analytics.exits), sub: "out of the room", badge: null },
  ];

  return (
    <div className="stagger grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      {tiles.map((t, i) => (
        <Card
          key={t.label}
          className="p-4 transition-colors hover:border-primary/40"
          style={{ "--i": i } as CSSProperties}
        >
          <div className="flex items-start justify-between gap-1">
            <div className="text-xs text-muted-foreground">{t.label}</div>
            {t.badge && (
              <Badge
                variant={t.badge.tone}
                className="px-1.5 py-0.5 text-[10px]"
                title="How confident the classifier is that this identity is the teacher"
              >
                {t.badge.label}
              </Badge>
            )}
          </div>
          <div className="mt-1.5 font-display text-2xl font-semibold tabular-nums tracking-tight">
            {t.value}
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">{t.sub}</div>
        </Card>
      ))}
    </div>
  );
}
