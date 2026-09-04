import type { RouterOutputs } from "@classroom/api-contracts";
import type { CSSProperties } from "react";
import { Card } from "@/components/ui/card";
import { Stat } from "@/components/ui/stat";
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
    sub: ms === null || ms === undefined ? absent : `${percentOf(ms, durationMs)} of the lesson`,
    badge: null as Confidence | null,
  });

  // Board time and the two actions; entries and exits live on the Attendance card.
  const tiles = [
    {
      ...duration("Time at board", analytics.teacherBoardMs, "no board zone"),
      badge,
    },
    duration("Pointing", analytics.teacherPointingMs, "not scored"),
    duration("Writing", analytics.teacherWritingMs, "not scored"),
  ];

  return (
    <div className="stagger grid gap-3 sm:grid-cols-3">
      {tiles.map((t, i) => (
        <Card key={t.label} className="p-4" style={{ "--i": i } as CSSProperties}>
          <Stat
            label={t.label}
            value={t.value}
            sub={t.badge ? `${t.sub} · ${t.badge.label} it is her` : t.sub}
            state={t.value === "n/a" ? "not_observed" : "observed"}
            reason={t.value === "n/a" ? t.sub : null}
          />
        </Card>
      ))}
    </div>
  );
}
