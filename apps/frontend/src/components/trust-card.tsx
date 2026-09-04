import type { RouterOutputs } from "@classroom/api-contracts";
import { ShieldCheck } from "lucide-react";
import { Card } from "@/components/ui/card";

type Trust = RouterOutputs["videos"]["get"]["trust"];

const STATE_LABEL = {
  observed: "Observed",
  provisional: "Provisional",
  not_observed: "Not Observed",
} as const;

const STATE_CLASS = {
  observed: "bg-tier-high/15 text-tier-high",
  provisional: "bg-tier-medium/15 text-tier-medium",
  not_observed: "bg-muted text-muted-foreground",
} as const;

/**
 * Group E — R22 and R23 over every measurement on the page: which numbers are
 * facts, which rest on phrase patterns until the labelling pass exists, and
 * which were withheld and why. The list is the requirements list itself, so
 * nothing is silently missing from it.
 */
export function TrustCard({ trust }: { trust: Trust }) {
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="flex items-center gap-2 font-display text-base font-semibold tracking-tight">
            <ShieldCheck className="size-4 text-muted-foreground" />
            Trust
          </h2>
          <p className="text-xs text-muted-foreground">
            R22 and R23: what could be seen and heard, and what is withheld rather than guessed.
          </p>
        </div>
        <div className="flex gap-1.5 text-[0.65rem]">
          <span className={`rounded-full px-2 py-0.5 ${STATE_CLASS.observed}`}>
            {trust.observed} observed
          </span>
          <span className={`rounded-full px-2 py-0.5 ${STATE_CLASS.provisional}`}>
            {trust.provisional} provisional
          </span>
          <span className={`rounded-full px-2 py-0.5 ${STATE_CLASS.not_observed}`}>
            {trust.notObserved} not observed
          </span>
        </div>
      </div>
      <ul className="mt-3 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
        {trust.items.map((item) => (
          <li key={item.id} className="flex items-baseline gap-2 py-0.5">
            <span className="w-8 shrink-0 font-mono text-[0.65rem] text-muted-foreground">
              {item.id}
            </span>
            <span className="min-w-0 flex-1">
              <span>{item.name}</span>
              {item.reason && (
                <span
                  className="block truncate text-[0.68rem] text-muted-foreground"
                  title={item.reason}
                >
                  {item.reason}
                </span>
              )}
            </span>
            <span
              className={`shrink-0 rounded-full px-1.5 py-0.5 text-[0.6rem] ${STATE_CLASS[item.state]}`}
            >
              {STATE_LABEL[item.state]}
            </span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
