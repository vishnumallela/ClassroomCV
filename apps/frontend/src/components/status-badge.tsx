import { Badge } from "@/components/ui/badge";

export function StatusBadge({ status }: { status: string }) {
  const variant = status === "done" ? "default" : status === "failed" ? "destructive" : "outline";
  return <Badge variant={variant}>{status}</Badge>;
}
