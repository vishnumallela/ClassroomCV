import { cn } from "@/lib/utils";

/**
 * One measurement, everywhere the same way: a plain-English label, the value,
 * one line of context, and its state. Three states, three words, one colour
 * each — read from a file (observed), rests on a phrase pattern until the
 * labelling pass exists (provisional), or withheld with a reason (not
 * observed). The requirement id, when given, is a tooltip, not a label.
 */

export type MeasureState = "observed" | "provisional" | "not_observed";

export const STATE_LABEL: Record<MeasureState, string> = {
  observed: "Observed",
  provisional: "Provisional",
  not_observed: "Not observed",
};

const STATE_CLASS: Record<MeasureState, string> = {
  observed: "bg-tier-high/15 text-tier-high",
  provisional: "bg-tier-medium/15 text-tier-medium",
  not_observed: "bg-muted text-muted-foreground",
};

export function StateChip({
  state,
  className,
  title,
}: {
  state: MeasureState;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex shrink-0 items-center rounded-full px-1.5 py-0.5 text-[0.6rem] font-medium leading-none",
        STATE_CLASS[state],
        className,
      )}
    >
      {STATE_LABEL[state]}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
  state,
  reason,
  id,
  size = "md",
  className,
}: {
  label: string;
  /** The number. "Not observed" is rendered from `state`, so pass what you have. */
  value: string;
  sub?: string | null;
  state?: MeasureState;
  /** Why, when the state is not observed; shown on hover. */
  reason?: string | null;
  /** Requirement id (R7…), shown as a tooltip on the label. */
  id?: string;
  size?: "md" | "lg";
  className?: string;
}) {
  const withheld = state === "not_observed";
  return (
    <div className={cn("min-w-0", className)} title={reason ?? undefined}>
      <div className="flex items-center justify-between gap-2">
        <span
          className="truncate text-xs text-muted-foreground"
          title={id ? `${id} · ${label}` : label}
        >
          {label}
        </span>
        {state && state !== "observed" && <StateChip state={state} />}
      </div>
      <div
        className={cn(
          "mt-1 font-display font-semibold tabular-nums tracking-tight",
          size === "lg" ? "text-2xl" : "text-lg",
          withheld && "text-muted-foreground",
        )}
      >
        {withheld ? "Not observed" : value}
      </div>
      {sub && <div className="mt-0.5 truncate text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}
