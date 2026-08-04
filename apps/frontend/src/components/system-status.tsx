import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { orpc } from "@/lib/orpc";

/**
 * Sidebar system block: the one-glance answer to "is the platform live?".
 * GPU pod + ML service state as a quiet LED readout, linking to Settings.
 */
export function SystemStatus() {
  const { data } = useQuery({
    ...orpc.gpu.status.queryOptions(),
    refetchInterval: 30_000,
    staleTime: 25_000,
  });

  const running = data?.pod?.desiredStatus === "RUNNING";
  const ready = running && data?.ml.healthy;
  const label = !data
    ? "checking…"
    : !data.configured
      ? "GPU not linked"
      : ready
        ? `GPU live · ${data.ml.device ?? "cuda"}`
        : running
          ? "GPU booting…"
          : "GPU standby";
  const led = !data
    ? "led-off"
    : ready
      ? "led-ok led-live"
      : running
        ? "led-warn led-live"
        : "led-off";

  return (
    <Link
      to="/settings"
      className="mx-1 mb-3 block rounded-lg border border-border/70 px-3 py-2.5 transition-colors hover:bg-accent"
    >
      <span className="micro-label block">System</span>
      <span className="mt-1.5 flex items-center gap-2">
        <span className={cn("led", led)} />
        <span className="font-mono text-xs text-foreground/90">{label}</span>
      </span>
    </Link>
  );
}
