import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Free-form pipeline statuses, mapped to operator-readable labels + an LED.
const LABEL: Record<string, string> = {
  waiting_gpu: "waiting for GPU",
};

function ledFor(status: string): string {
  if (status === "done") return "led-ok";
  if (status === "failed") return "bg-destructive led";
  if (status === "waiting_gpu") return "led-off";
  return "led-warn led-live"; // queued / probing / analyzing / deriving
}

export function StatusBadge({ status }: { status: string }) {
  const variant = status === "done" ? "default" : status === "failed" ? "destructive" : "outline";
  return (
    <Badge
      variant={variant}
      className="gap-1.5 font-mono text-[0.62rem] uppercase tracking-[0.1em]"
    >
      <span className={cn("led", ledFor(status))} />
      {LABEL[status] ?? status}
    </Badge>
  );
}
