import type { RouterOutputs } from "@classroom/api-contracts";
import { Footprints, Info } from "lucide-react";
import { Card } from "@/components/ui/card";
import { type Heatmap, circulation } from "@/lib/analytics";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;

const pct = (n: number) => `${Math.round(n * 100)}%`;

/**
 * Teacher circulation, derived from the dwell heatmap: how much of the room the
 * teacher's path covered, how much of that time was spent among the desks
 * (proximity is linked to on-task behaviour in the research), and whether she
 * reached the back rows. Deliberately framed as IMAGE-PLANE coverage — never
 * distance in feet, which we cannot honestly measure without calibration.
 */
export function CirculationCard({ analytics }: { analytics: Analytics }) {
  const c = circulation(analytics.heatmap as Heatmap);
  if (!c) return null;

  const summary =
    c.focusShare > 0.5
      ? "Mostly taught from one spot"
      : c.reachedBackRows
        ? "Circulated widely, reaching the back rows"
        : "Circulated near the front of the room";

  const tiles = [
    {
      label: "Room covered",
      value: pct(c.coverage),
      hint: "Share of the visible room the teacher's path touched.",
    },
    {
      label: "Time among desks",
      value: pct(c.amongStudentsShare),
      hint: "Share of teaching time spent where students sit.",
    },
    {
      label: "Spent in one spot",
      value: pct(c.focusShare),
      hint: "Share of time in her single most-used position (lower = more mobile).",
    },
    {
      label: "Reached back rows",
      value: c.reachedBackRows ? "Yes" : "No",
      hint: "Did the teacher enter the row band farthest from the front.",
    },
  ];

  return (
    <Card className="p-5">
      <div className="flex items-center gap-2">
        <Footprints className="size-4 text-primary" />
        <h2 className="text-sm font-medium">Teacher circulation</h2>
      </div>
      <p className="mt-1 font-display text-lg tracking-tight">{summary}</p>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {tiles.map((t) => (
          <div key={t.label} title={t.hint}>
            <div className="font-display text-2xl font-semibold tabular-nums">{t.value}</div>
            <div className="mt-0.5 text-xs text-muted-foreground">{t.label}</div>
          </div>
        ))}
      </div>

      <p className="mt-4 flex items-start gap-1.5 border-t border-border pt-3 text-[0.7rem] leading-relaxed text-muted-foreground">
        <Info className="mt-px size-3 shrink-0" />
        Coverage is measured on the camera image, not in feet. The dwell map below shows the same
        path in detail.
      </p>
    </Card>
  );
}
