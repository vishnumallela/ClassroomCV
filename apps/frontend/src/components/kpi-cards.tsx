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
  const badge = confidenceBadge(teacherConfidence);

  /**
   * A duration KPI that can be genuinely unknown.
   *
   * null is not zero and must never render as "0:00": board time is null until
   * a board zone exists, and the action KPIs are null on a lesson analysed
   * before they shipped or re-derived from teacher-only stored rows. Showing
   * 0:00 there would assert she never did it.
   */
  const duration = (label: string, ms: number | null | undefined, absent: string) => ({
    label,
    value: ms === null || ms === undefined ? "n/a" : msToClock(ms),
    sub: ms === null || ms === undefined ? absent : `${percentOf(ms, durationMs)} of lesson`,
    badge: null as Confidence | null,
  });

  // Durations first, then the two counts, so the grid breaks between the two
  // kinds of number rather than mid-group.
  const tiles = [
    {
      ...duration("Time at board", analytics.teacherBoardMs, "no board zone"),
      badge,
    },
    duration("Pointing", analytics.teacherPointingMs, "not scored"),
    duration("Writing", analytics.teacherWritingMs, "not scored"),
    {
      label: "Teacher entries",
      value: String(analytics.entries),
      sub: "into the room",
      badge: null,
    },
    {
      label: "Teacher exits",
      value: String(analytics.exits),
      sub: "out of the room",
      badge: null,
    },
  ];

  return (
    <div className="stagger grid grid-cols-3 gap-3">
      {tiles.map((t, i) => (
        <Card
          key={t.label}
          className="p-4 transition-colors hover:border-primary/40"
          style={{ "--i": i } as CSSProperties}
        >
          <div className="flex items-start justify-between gap-1">
            <div className="micro-label">{t.label}</div>
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
          <div className="mt-2 font-mono text-2xl font-semibold tabular-nums tracking-tight">
            {t.value}
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">{t.sub}</div>
        </Card>
      ))}
    </div>
  );
}
