import type { RouterOutputs } from "@classroom/api-contracts";
import { Footprints, Info } from "lucide-react";
import { Card } from "@/components/ui/card";
import { type Heatmap, circulation } from "@/lib/analytics";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;

const pct = (n: number) => `${Math.round(n * 100)}%`;

// Movement-pattern label (Moodoo). Describes WHERE the teacher taught from, not
// how well, never a quality judgement.
const STYLE_LABEL: Record<string, { title: string; blurb: string }> = {
  presenter: {
    title: "Front-of-room presenter",
    blurb: "Taught mostly from one or two positions near the front.",
  },
  supervisor: {
    title: "Circulating supervisor",
    blurb: "Moved widely through the room and reached the back rows.",
  },
  balanced: {
    title: "Balanced circulation",
    blurb: "Mixed time at the front with movement among the desks.",
  },
};

/**
 * Teacher circulation, derived from the dwell heatmap. Reports how much of the
 * room the path covered, how spread the dwell was, and a Moodoo-style movement
 * pattern (presenter vs supervisor). Everything is IMAGE-PLANE and relative to
 * this lesson, never distance in feet (needs camera calibration) and never a
 * judgement of teaching quality.
 */
export function CirculationCard({ analytics }: { analytics: Analytics }) {
  const c = circulation(analytics.heatmap as Heatmap);
  if (!c) return null;
  const style = STYLE_LABEL[c.style] ?? STYLE_LABEL.balanced!;

  const tiles = [
    {
      label: "Room covered",
      value: pct(c.coverage),
      hint: "Share of the visible room the teacher's path touched.",
    },
    {
      label: "Movement spread",
      value: pct(c.spread),
      hint: "How evenly the teacher's time was distributed across the room (0% = one spot, 100% = even). A validated mobility measure (Moodoo).",
    },
    {
      label: "Time among desks",
      value: pct(c.amongStudentsShare),
      hint: "Share of teaching time spent where students sit.",
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
      <p className="mt-1 font-display text-lg tracking-tight">{style.title}</p>
      <p className="text-sm text-muted-foreground">{style.blurb}</p>

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
        A movement pattern, not a rating: it describes where the teacher taught from, measured on
        the camera image (not in feet). The dwell map below shows the same path in detail.
      </p>
    </Card>
  );
}
