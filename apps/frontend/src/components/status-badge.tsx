import { Badge } from "@/components/ui/badge";

// Free-form pipeline statuses, mapped to operator-readable labels.
const LABEL: Record<string, string> = {
  waiting_gpu: "waiting for GPU",
};

export function StatusBadge({ status }: { status: string }) {
  const variant = status === "done" ? "default" : status === "failed" ? "destructive" : "outline";
  return <Badge variant={variant}>{LABEL[status] ?? status}</Badge>;
}
