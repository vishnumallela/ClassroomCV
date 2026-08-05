import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Cpu, Play, Power, Server } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { orpc, orpcClient } from "@/lib/orpc";

export const Route = createFileRoute("/settings")({ component: Settings });

const POD_STATE: Record<string, { label: string; tone: "high" | "medium" | "low" }> = {
  RUNNING: { label: "Running", tone: "high" },
  EXITED: { label: "Stopped", tone: "medium" },
  TERMINATED: { label: "Terminated", tone: "low" },
};

/**
 * Application settings: the RunPod GPU (an on-demand pod, deliberately not
 * serverless — the TensorRT engine lives on its volume) and where the API
 * finds the ML service. The stop button is the cost lever: a stopped pod
 * bills volume storage only.
 */
function Settings() {
  const queryClient = useQueryClient();
  const settings = useQuery(orpc.settings.get.queryOptions());
  const gpu = useQuery({
    ...orpc.gpu.status.queryOptions(),
    refetchInterval: 10_000,
  });

  const [apiKey, setApiKey] = useState("");
  const [podId, setPodId] = useState<string | null>(null);
  const [mlUrl, setMlUrl] = useState<string | null>(null);
  const [autoStart, setAutoStart] = useState<boolean | null>(null);
  const [autoStopMin, setAutoStopMin] = useState<number | null>(null);
  const [saved, setSaved] = useState(false);

  const save = useMutation({
    mutationFn: () =>
      orpcClient.settings.update({
        ...(apiKey ? { runpodApiKey: apiKey } : {}),
        ...(podId !== null ? { runpodPodId: podId } : {}),
        ...(mlUrl !== null ? { mlServiceUrl: mlUrl } : {}),
        ...(autoStart !== null ? { gpuAutoStart: autoStart } : {}),
        ...(autoStopMin !== null ? { gpuAutoStopMinutes: autoStopMin } : {}),
      }),
    onSuccess: async () => {
      setApiKey("");
      setPodId(null);
      setMlUrl(null);
      setAutoStart(null);
      setAutoStopMin(null);
      setSaved(true);
      await queryClient.invalidateQueries();
    },
  });

  const start = useMutation({
    mutationFn: () => orpcClient.gpu.start(),
    onSuccess: () => queryClient.invalidateQueries(),
  });
  const stop = useMutation({
    mutationFn: () => orpcClient.gpu.stop(),
    onSuccess: () => queryClient.invalidateQueries(),
  });

  if (settings.isLoading) {
    return <Skeleton className="h-72 rounded-xl" />;
  }
  const s = settings.data;
  const g = gpu.data;
  // Transitional/unknown states from RunPod still ARE an answer — show them
  // verbatim instead of pretending the pod is unreachable.
  const podState = g?.pod
    ? (POD_STATE[g.pod.desiredStatus] ?? {
        label: g.pod.desiredStatus.toLowerCase(),
        tone: "medium" as const,
      })
    : null;
  const dirty =
    apiKey !== "" || podId !== null || mlUrl !== null || autoStart !== null || autoStopMin !== null;

  return (
    <div className="space-y-8">
      <header className="reveal space-y-1.5">
        <span className="micro-label">System</span>
        <h1 className="font-display text-3xl font-semibold tracking-tight">Settings</h1>
        <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">
          The GPU that runs analysis, and how this app reaches it. Lessons queue while the GPU is
          off and process when it comes back.
        </p>
      </header>

      {/* ---- GPU control ---------------------------------------------------- */}
      <Card className="space-y-5 p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <Cpu className="size-4 text-primary" />
            <h2 className="font-display text-lg font-medium">GPU worker (RunPod)</h2>
          </div>
          {g?.configured === false ? (
            <Badge variant="outline">not configured</Badge>
          ) : podState ? (
            <Badge variant={podState.tone}>{podState.label}</Badge>
          ) : gpu.isLoading ? (
            <Badge variant="outline">checking…</Badge>
          ) : (
            <Badge variant="low">unreachable</Badge>
          )}
        </div>

        {g?.configured === false ? (
          <p className="text-sm text-muted-foreground">
            Add your RunPod API key and pod id below to control the GPU from here.
          </p>
        ) : (
          <>
            <div className="grid gap-3 text-sm sm:grid-cols-3">
              <div>
                <div className="text-xs text-muted-foreground">Pod</div>
                <div className="mt-0.5 font-medium">{g?.pod?.name ?? g?.pod?.id ?? "—"}</div>
                <div className="text-xs text-muted-foreground">{g?.pod?.gpuTypeId ?? ""}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Billing while running</div>
                <div className="mt-0.5 font-medium tabular-nums">
                  {g?.pod?.costPerHr != null ? `$${g.pod.costPerHr.toFixed(2)}/hr` : "—"}
                </div>
                <div className="text-xs text-muted-foreground">stopped pods bill storage only</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">ML service</div>
                <div className="mt-0.5 font-medium">
                  {g?.ml.healthy ? `healthy · ${g.ml.device ?? "?"}` : "unreachable"}
                </div>
                <div className="truncate text-xs text-muted-foreground">
                  {g?.ml.model ? g.ml.model.split("/").pop() : ""}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                size="sm"
                disabled={start.isPending || g?.pod?.desiredStatus === "RUNNING"}
                onClick={() => start.mutate()}
              >
                <Play className="size-3.5" />
                {start.isPending ? "Starting…" : "Start GPU"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={stop.isPending || g?.pod?.desiredStatus !== "RUNNING"}
                onClick={() => stop.mutate()}
              >
                <Power className="size-3.5" />
                {stop.isPending ? "Stopping…" : "Stop GPU"}
              </Button>
              {(start.isError || stop.isError || g?.podError) && (
                <span className="text-xs text-destructive">
                  {g?.podError ?? "RunPod call failed."}
                </span>
              )}
            </div>
          </>
        )}
      </Card>

      {/* ---- Configuration --------------------------------------------------- */}
      <Card className="space-y-4 p-6">
        <div className="flex items-center gap-2">
          <Server className="size-4 text-primary" />
          <h2 className="font-display text-lg font-medium">Connection</h2>
        </div>
        <form
          className="grid gap-4 sm:grid-cols-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (dirty) save.mutate();
          }}
        >
          <label className="block space-y-1.5">
            <span className="text-sm font-medium">RunPod API key</span>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setSaved(false);
              }}
              placeholder={s?.runpodApiKeyMasked ?? "rpa_…"}
              autoComplete="off"
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
            <span className="block text-xs text-muted-foreground">
              Stored server-side, shown masked. Leave blank to keep the current key.
            </span>
          </label>
          <label className="block space-y-1.5">
            <span className="text-sm font-medium">RunPod pod ID</span>
            <input
              value={podId ?? s?.runpodPodId ?? ""}
              onChange={(e) => {
                setPodId(e.target.value);
                setSaved(false);
              }}
              placeholder="abc123xyz"
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
            <span className="block text-xs text-muted-foreground">
              The on-demand GPU pod (not serverless) created from the ml-service image.
            </span>
          </label>
          <label className="flex items-start gap-3 rounded-lg border border-border p-3">
            <input
              type="checkbox"
              checked={autoStart ?? s?.gpuAutoStart ?? false}
              onChange={(e) => {
                setAutoStart(e.target.checked);
                setSaved(false);
              }}
              className="mt-0.5 size-4 accent-primary"
            />
            <span className="space-y-0.5">
              <span className="block text-sm font-medium">Auto-start GPU</span>
              <span className="block text-xs text-muted-foreground">
                Start the pod automatically when a lesson is queued and the GPU is off.
              </span>
            </span>
          </label>
          <label className="block space-y-1.5">
            <span className="text-sm font-medium">Auto-stop after idle (minutes)</span>
            <input
              type="number"
              min={0}
              max={1440}
              value={autoStopMin ?? s?.gpuAutoStopMinutes ?? 0}
              onChange={(e) => {
                setAutoStopMin(Math.max(0, Math.floor(Number(e.target.value) || 0)));
                setSaved(false);
              }}
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
            <span className="block text-xs text-muted-foreground">
              Stop the pod once the queue has been empty this long. 0 disables.
            </span>
          </label>
          <label className="block space-y-1.5 sm:col-span-2">
            <span className="text-sm font-medium">ML service URL</span>
            <input
              value={mlUrl ?? s?.mlServiceUrl ?? ""}
              onChange={(e) => {
                setMlUrl(e.target.value);
                setSaved(false);
              }}
              placeholder={
                s?.mlServiceUrlEffective ?? s?.mlServiceUrlDefault ?? "http://127.0.0.1:8000"
              }
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
            <span className="block text-xs text-muted-foreground">
              Leave this empty. With a pod ID set, the URL is derived from it — RunPod's proxy
              hostname is the one address that survives stop/start, so autopilot never leaves a
              stale URL behind. Fill it in only to override (a tunnel, a second pod); changes apply
              immediately, no redeploy.
            </span>
          </label>
          <div className="flex items-center gap-3 sm:col-span-2">
            <Button type="submit" size="sm" disabled={!dirty || save.isPending}>
              {save.isPending ? "Saving…" : "Save settings"}
            </Button>
            {saved && !dirty && <span className="text-xs text-muted-foreground">Saved.</span>}
            {save.isError && (
              <span className="text-xs text-destructive">Save failed. Check the values.</span>
            )}
          </div>
        </form>
      </Card>
    </div>
  );
}
