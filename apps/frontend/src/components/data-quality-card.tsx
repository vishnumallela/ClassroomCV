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

function TierPill({ tier, size = "sm" }: { tier: Tier; size?: "sm" | "lg" }) {
  const m = TIER_META[tier];
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
    label: "Camera coverage",
    help: "How much of the lesson the camera actually saw people.",
  },
  {
    key: "identity",
    label: "Tracking",
    help: "How cleanly people were followed without fragmenting.",
  },
  { key: "occupancy", label: "Head counts", help: "How reliable the student counts are." },
  { key: "teacher", label: "Teacher ID", help: "How confidently the teacher was identified." },
];

export function DataQualityCard({ analytics }: { analytics: Analytics }) {
  const dq = analytics.dataQuality;
  if (!dq) return null;
  const overall = dq.confidence.overall as Tier;

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-4 text-primary" />
          <h2 className="text-sm font-medium">How reliable are these numbers?</h2>
        </div>
        <TierPill tier={overall} size="lg" />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {DIMENSIONS.map((d) => (
          <div
            key={d.key}
            className="rounded-lg border border-border bg-background/50 p-3"
            title={d.help}
          >
            <div className="text-xs text-muted-foreground">{d.label}</div>
            <div className="mt-1.5">
              <TierPill tier={dq.confidence[d.key] as Tier} />
            </div>
          </div>
        ))}
      </div>

      {/* Re-identification-independent cross-check on the crowd count. */}
      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg bg-muted/50 px-4 py-3 text-sm">
        <CrossCheck
          label="Seen at once"
          value={dq.concurrent_peak}
          hint="Most people visible in a single frame (can't double-count anyone)."
        />
        <CrossCheck
          label="Distinct identities"
          value={dq.identities}
          hint="How many separate people the tracker resolved."
        />
        <CrossCheck
          label="Camera coverage"
          value={`${Math.round(dq.coverage * 100)}%`}
          hint="Share of the active lesson with a person in view."
        />
        <CrossCheck
          label="Fragments / person"
          value={dq.fragmentation.toFixed(1)}
          hint="Tracker ids merged into each identity (1.0 is perfect)."
        />
      </div>

      {dq.notes.length > 0 && (
        <ul className="mt-4 space-y-1.5">
          {dq.notes.map((note) => (
            <li key={note} className="flex gap-2 text-xs leading-relaxed text-muted-foreground">
              <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-tier-medium" />
              <span>{note}</span>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-4 flex items-start gap-1.5 border-t border-border pt-3 text-[0.7rem] leading-relaxed text-muted-foreground">
        <Info className="mt-px size-3 shrink-0" />
        Aggregate estimates from video sampled at 5 frames per second. Head counts are a proxy, not
        an attendance register. No faces are recognized and no student is named.
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
