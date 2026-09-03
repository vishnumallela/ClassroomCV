import type { RouterOutputs } from "@classroom/api-contracts";
import { CircleAlert, Info, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;
type DataQuality = NonNullable<Analytics["dataQuality"]>;
type Tier = "high" | "medium" | "low";

const TIER_META: Record<Tier, { label: string; dot: string }> = {
  high: { label: "Strong", dot: "bg-tier-high" },
  medium: { label: "Fair", dot: "bg-tier-medium" },
  low: { label: "Tentative", dot: "bg-tier-low" },
};

// `data_quality` is stored jsonb, so rows outlive the code that wrote them.
// Videos analysed by the previous pipeline carry a different shape (identities /
// raw_tracks / fragmentation, and no `continuity` tier), and reading a missing
// field off one of those used to throw and blank the whole page. Every accessor
// below therefore tolerates absence and renders an em dash.
function TierPill({ tier, size = "sm" }: { tier?: Tier; size?: "sm" | "lg" }) {
  const m = tier ? TIER_META[tier] : undefined;
  if (!m || !tier) {
    return (
      <Badge variant="low" className={size === "lg" ? "px-2.5 py-1 text-xs" : "text-[0.7rem]"}>
        —
      </Badge>
    );
  }
  return (
    <Badge variant={tier} className={size === "lg" ? "px-2.5 py-1 text-xs" : "text-[0.7rem]"}>
      <span className={cn("size-1.5 rounded-full", m.dot)} />
      {m.label}
    </Badge>
  );
}

const DIMENSIONS: { key: keyof DataQuality["confidence"]; label: string; help: string }[] = [
  {
    key: "coverage",
    label: "Teacher coverage",
    help: "How much of the lesson the teacher was actually visible for.",
  },
  {
    key: "continuity",
    label: "Continuity",
    help: "How often her timeline broke. Entries and exits are counted from those breaks.",
  },
  {
    key: "teacher",
    label: "Detection",
    help: "How confident the detector was when it found her.",
  },
  {
    key: "attribution",
    label: "Attribution",
    help:
      "Whether there was only one adult to follow. The other three describe how well one " +
      "person was tracked; this one asks whether tracking one person was right at all.",
  },
];

export function DataQualityCard({ analytics }: { analytics: Analytics }) {
  const dq = analytics.dataQuality;
  if (!dq) return null;
  const overall = dq.confidence?.overall as Tier | undefined;
  const num = (v: unknown, digits = 0): string =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
  const pct = (v: unknown): string =>
    typeof v === "number" && Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—";

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-4 text-primary" />
          <h2 className="text-sm font-medium">How reliable are these numbers?</h2>
        </div>
        <TierPill tier={overall} size="lg" />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {DIMENSIONS.map((d) => (
          <div
            key={d.key}
            className="rounded-lg border border-border bg-background/50 p-3"
            title={d.help}
          >
            <div className="text-xs text-muted-foreground">{d.label}</div>
            <div className="mt-1.5">
              <TierPill tier={dq.confidence?.[d.key] as Tier | undefined} />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg bg-muted/50 px-4 py-3 text-sm">
        <CrossCheck
          label="Teacher coverage"
          value={pct(dq.coverage)}
          hint="Share of sampled frames the teacher was found in."
        />
        <CrossCheck
          label="Timeline breaks"
          value={num(dq.breaks)}
          hint="Gaps in her timeline. Entries and exits are counted from these."
        />
        <CrossCheck
          label="Detection confidence"
          value={num(dq.mean_confidence, 2)}
          hint="Average score of the detections behind her timeline."
        />
        {dq.multiple_adults_detected === true && (
          <CrossCheck
            label="Adults at once"
            value={num(dq.max_simultaneous_adults)}
            hint="The most adults the detector saw in the room at the same moment."
          />
        )}
      </div>

      {(dq.notes ?? []).length > 0 && (
        <ul className="mt-4 space-y-1.5">
          {(dq.notes ?? []).map((note) => (
            <li key={note} className="flex gap-2 text-xs leading-relaxed text-muted-foreground">
              <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-tier-medium" />
              <span>{note}</span>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-4 flex items-start gap-1.5 border-t border-border pt-3 text-[0.7rem] leading-relaxed text-muted-foreground">
        <Info className="mt-px size-3 shrink-0" />
        Aggregate estimates from video sampled at 5 frames per second. No faces are recognized and
        no student is detected at all — the model looks only for the teacher, the board and the
        door.
      </p>
    </Card>
  );
}

function CrossCheck({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | number;
  hint: string;
}) {
  return (
    <div title={hint}>
      <div className="font-display text-lg font-semibold tabular-nums leading-none">{value}</div>
      <div className="mt-1 text-xs text-muted-foreground">{label}</div>
    </div>
  );
}
