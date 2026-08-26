import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { AlertTriangle, Cpu, HardDrive, Play, Power, Server, Trash2, Zap } from "lucide-react";
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

const FIELD_CLASS =
  "w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25";

function Field({
  label,
  hint,
  children,
  className = "",
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <label className={`block space-y-1.5 ${className}`}>
      <span className="text-sm font-medium">{label}</span>
      {children}
      {hint && <span className="block text-xs leading-relaxed text-muted-foreground">{hint}</span>}
    </label>
  );
}

function Check({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3 rounded-lg border border-border p-3">
      <input
        type="checkbox"
        aria-label={label}
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 size-4 accent-primary"
      />
      <span className="space-y-0.5">
        <span className="block text-sm font-medium">{label}</span>
        <span className="block text-xs leading-relaxed text-muted-foreground">{hint}</span>
      </span>
    </label>
  );
}

/**
 * Settings: the RunPod GPU worker, provisioned and destroyed from here.
 *
 * The point of this page is that the RunPod console is never needed. Every
 * field the console would ask for — GPU, cloud tier, region, network volume,
 * image, container env — is a control here, backed by RunPod's live catalog, so
 * a pod can be created, watched and terminated without leaving the app. Cost
 * control follows from that: a terminated pod bills nothing but its volume, and
 * the checkpoint survives on the volume, so the replacement comes up ready.
 */
function Settings() {
  const queryClient = useQueryClient();
  const settings = useQuery(orpc.settings.get.queryOptions());
  const gpu = useQuery({ ...orpc.gpu.status.queryOptions(), refetchInterval: 10_000 });
  const catalog = useQuery(orpc.gpu.catalog.queryOptions());
  const imageCheck = useQuery(orpc.gpu.image.queryOptions());

  // One draft bag rather than a useState per field: the pod spec is ~20 fields
  // and they are saved together.
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);
  const [volName, setVolName] = useState("classroomcv-weights");
  const [volSize, setVolSize] = useState("100");

  const set = (key: string, value: string) => {
    setDraft((d) => ({ ...d, [key]: value }));
    setSaved(false);
  };
  const dirty = Object.keys(draft).length > 0;

  const s = settings.data;
  const g = gpu.data;
  const spec = s?.spec;
  /** Draft value if edited, else the saved/resolved one. */
  const val = (key: string, current: string | number | null | undefined) =>
    draft[key] ?? String(current ?? "");
  const flag = (key: string, current: boolean | undefined) =>
    key in draft ? draft[key] === "true" : (current ?? false);

  const save = useMutation({
    mutationFn: () => {
      const num = (k: string) => (k in draft ? Number(draft[k]) || 0 : undefined);
      const str = (k: string) => draft[k];
      const bool = (k: string) => (k in draft ? draft[k] === "true" : undefined);
      return orpcClient.settings.update({
        ...(draft.runpodApiKey ? { runpodApiKey: draft.runpodApiKey } : {}),
        ...def("runpodPodId", str("runpodPodId")),
        ...def("mlServiceUrl", str("mlServiceUrl")),
        ...def("gpuAutoStart", bool("gpuAutoStart")),
        ...def("gpuAutoStopMinutes", num("gpuAutoStopMinutes")),
        ...def("gpuIdleAction", str("gpuIdleAction") as "terminate" | "stop" | undefined),
        ...def("gpuPodName", str("gpuPodName")),
        ...def("gpuImage", str("gpuImage")),
        ...def("gpuTypeId", str("gpuTypeId")),
        ...def("gpuCount", num("gpuCount")),
        ...def("gpuCloudType", str("gpuCloudType") as "SECURE" | "COMMUNITY" | undefined),
        ...def("gpuDataCenterId", str("gpuDataCenterId")),
        ...def("gpuNetworkVolumeId", str("gpuNetworkVolumeId")),
        ...def("gpuVolumeMountPath", str("gpuVolumeMountPath")),
        ...def("gpuContainerDiskGb", num("gpuContainerDiskGb")),
        ...def("gpuCudaVersions", str("gpuCudaVersions")),
        ...def("gpuInterruptible", bool("gpuInterruptible")),
        ...def("gpuSshPublicKey", str("gpuSshPublicKey")),
        ...def("mlWeightsPath", str("mlWeightsPath")),
        ...def("mlBatch", num("mlBatch")),
        ...def("mlResolution", num("mlResolution")),
        ...def("mlTensorrt", bool("mlTensorrt")),
        ...def("mlMediaAllowlist", str("mlMediaAllowlist")),
        ...def("mlDatabaseUrl", str("mlDatabaseUrl")),
      });
    },
    onSuccess: async () => {
      setDraft({});
      setSaved(true);
      await queryClient.invalidateQueries();
    },
  });

  const act = (fn: () => Promise<unknown>) => ({
    mutationFn: fn,
    onSuccess: () => queryClient.invalidateQueries(),
  });
  const create = useMutation(act(() => orpcClient.gpu.create()));
  const terminate = useMutation(act(() => orpcClient.gpu.terminate()));
  const start = useMutation(act(() => orpcClient.gpu.start()));
  const stop = useMutation(act(() => orpcClient.gpu.stop()));
  const createVolume = useMutation(
    act(() =>
      orpcClient.gpu.createVolume({
        name: volName,
        sizeGb: Number(volSize) || 100,
        dataCenterId: val("gpuDataCenterId", spec?.dataCenterId),
      }),
    ),
  );

  if (settings.isLoading || !s || !spec) {
    return <Skeleton className="h-72 rounded-xl" />;
  }

  // Transitional/unknown states from RunPod still ARE an answer — show them
  // verbatim instead of pretending the pod is unreachable.
  const podState = g?.pod
    ? (POD_STATE[g.pod.desiredStatus] ?? {
        label: g.pod.desiredStatus.toLowerCase(),
        tone: "medium" as const,
      })
    : null;
  const hasPod = Boolean(g?.pod);
  const cloud = val("gpuCloudType", spec.cloudType) as "SECURE" | "COMMUNITY";
  const dcId = val("gpuDataCenterId", spec.dataCenterId);
  const volumeId = val("gpuNetworkVolumeId", spec.networkVolumeId);
  const cat = catalog.data;

  /** The price on the tier currently selected — the one shown in the option. */
  const priceOn = (x: { price: { secure: number; community: number } }) =>
    cloud === "SECURE" ? x.price.secure : x.price.community;

  // Only cards actually purchasable on the selected tier. A GPU with no stock
  // there fails at create time, after the user has committed to the choice.
  // Re-sorted per tier, not left in the server's order: the server sorts by
  // community price, so a Secure list ordered by it reads as arbitrary.
  const gpus = (cat?.gpus ?? [])
    .filter((x) => (cloud === "SECURE" ? x.maxCount.secure : x.maxCount.community) > 0)
    .toSorted((a, b) => priceOn(a) - priceOn(b));
  const chosenGpu = gpus.find((x) => x.id === val("gpuTypeId", spec.gpuTypeId));
  const rate = chosenGpu ? priceOn(chosenGpu) : null;
  // A volume is pinned to its region for life and a pod can only mount one in
  // its own region, so the list is region-scoped rather than merely sorted.
  const volumesHere = (cat?.volumes ?? []).filter((v) => v.dataCenterId === dcId);

  // What the configured tag ACTUALLY contains, read from its registry. A tag
  // is not a description: `:latest` here is written only by main, so it can be
  // an entirely different program with the same name and the same /health.
  const img = imageCheck.data?.check;
  const imgInfo = img?.status === "ok" ? img.info : null;

  const blockers: string[] = [];
  if (!s.runpodApiKeyMasked) blockers.push("Add a RunPod API key.");
  if (img?.status === "not-found") {
    blockers.push(`${img.message} A pod on it would boot, fail to pull, and bill anyway.`);
  }
  if (imgInfo && !imgInfo.isMlService) {
    // Name what it declares INSTEAD — that is how you recognise the old YOLO
    // image (MODEL_NAME, IMGSZ) at a glance versus something else entirely.
    // DEVICE/REQUIRE_DEVICE are dropped because both images carry them, so
    // they distinguish nothing and only crowd out the informative names.
    const declared =
      Object.keys(imgInfo.env)
        .filter((k) => k !== "DEVICE" && k !== "REQUIRE_DEVICE")
        .slice(0, 4)
        .join(", ") || "nothing recognisable";
    blockers.push(
      `That image is not the RF-DETR ML service — it declares ${declared}, not RFDETR_WEIGHTS. ` +
        "Every RFDETR_* setting below would be ignored by it.",
    );
  }

  const warnings: string[] = [];
  if (!volumeId) {
    // Not a blocker: RunPod refuses to create a network volume below a $5
    // account balance, and an account that cannot do that can still rent a GPU.
    warnings.push(
      `No network volume in ${dcId} — the pod gets its own disk instead, which dies with it. ` +
        "The checkpoint must be re-uploaded to every new pod (~40s for 255 MB).",
    );
  }
  if (!spec.env.MEDIA_URL_ALLOWLIST) {
    warnings.push("MEDIA_URL_ALLOWLIST is unset — the pod will refuse to fetch any video.");
  }
  if (!s.mlDatabaseUrlMasked) {
    warnings.push("DATABASE_URL is unset — the pod cannot write detections back.");
  }
  if (!s.sshPublicKeySet) {
    warnings.push("No SSH key — you will not be able to upload the checkpoint to a fresh volume.");
  }

  return (
    <div className="space-y-8">
      <header className="reveal space-y-1.5">
        <span className="micro-label">System</span>
        <h1 className="font-display text-3xl font-semibold tracking-tight">Settings</h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          The GPU that runs analysis — specified, created and destroyed from here. Nothing on this
          page needs the RunPod console. Lessons queue while the GPU is down and process when it
          comes back.
        </p>
      </header>

      {/* ---- GPU control ---------------------------------------------------- */}
      <Card className="space-y-5 p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <Cpu className="size-4 text-primary" />
            <h2 className="font-display text-lg font-medium">GPU worker</h2>
          </div>
          {g?.configured === false ? (
            <Badge variant="outline">no pod</Badge>
          ) : podState ? (
            <Badge variant={podState.tone}>{podState.label}</Badge>
          ) : gpu.isLoading ? (
            <Badge variant="outline">checking…</Badge>
          ) : (
            <Badge variant="low">unreachable</Badge>
          )}
        </div>

        {hasPod ? (
          <div className="grid gap-3 text-sm sm:grid-cols-4">
            <div>
              <div className="text-xs text-muted-foreground">Pod</div>
              <div className="mt-0.5 font-medium">{g?.pod?.name ?? g?.pod?.id ?? "—"}</div>
              <div className="truncate text-xs text-muted-foreground">{g?.pod?.id}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">GPU</div>
              <div className="mt-0.5 font-medium">{g?.pod?.gpuTypeId ?? "—"}</div>
              <div className="text-xs text-muted-foreground">{g?.pod?.dataCenterId ?? ""}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Billing while running</div>
              <div className="mt-0.5 font-medium tabular-nums">
                {typeof g?.pod?.costPerHr === "number" ? `$${g.pod.costPerHr.toFixed(2)}/hr` : "—"}
              </div>
              <div className="text-xs text-muted-foreground">volume bills either way</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">ML service</div>
              <div className="mt-0.5 font-medium">
                {g?.ml.healthy ? `healthy · ${g.ml.device ?? "?"}` : "unreachable"}
              </div>
              <div className="truncate text-xs text-muted-foreground">
                {g?.ml.model ? g.ml.model.split("/").pop() : ""}
                {g?.ml.backend ? ` · ${g.ml.backend}` : ""}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No pod running. Creating one takes the spec below — GPU, region, volume, image — and
            provisions it directly. It costs{" "}
            {rate === null ? "the selected card's hourly rate" : `$${rate.toFixed(2)}/hr`} until it
            is terminated.
          </p>
        )}

        {(blockers.length > 0 || warnings.length > 0) && (
          <div className="space-y-1.5 rounded-lg border border-border bg-muted/30 p-3">
            {blockers.map((b) => (
              <div key={b} className="flex items-start gap-2 text-xs text-destructive">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{b}</span>
              </div>
            ))}
            {warnings.map((w) => (
              <div key={w} className="flex items-start gap-2 text-xs text-muted-foreground">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{w}</span>
              </div>
            ))}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2">
          {!hasPod && (
            <Button
              size="sm"
              disabled={create.isPending || blockers.length > 0 || dirty}
              onClick={() => create.mutate()}
            >
              <Zap className="size-3.5" />
              {create.isPending ? "Provisioning…" : "Create pod"}
            </Button>
          )}
          {hasPod && (
            <>
              <Button
                size="sm"
                disabled={start.isPending || g?.pod?.desiredStatus === "RUNNING"}
                onClick={() => start.mutate()}
              >
                <Play className="size-3.5" />
                {start.isPending ? "Starting…" : "Start"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={stop.isPending || g?.pod?.desiredStatus !== "RUNNING"}
                onClick={() => stop.mutate()}
              >
                <Power className="size-3.5" />
                {stop.isPending ? "Stopping…" : "Stop"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={terminate.isPending}
                onClick={() => terminate.mutate()}
              >
                <Trash2 className="size-3.5" />
                {terminate.isPending ? "Terminating…" : "Terminate"}
              </Button>
            </>
          )}
          {dirty && !hasPod && (
            <span className="text-xs text-muted-foreground">Save the spec before creating.</span>
          )}
          {(create.error || terminate.error || start.error || stop.error || g?.podError) && (
            <span className="text-xs text-destructive">
              {create.error?.message ??
                terminate.error?.message ??
                start.error?.message ??
                stop.error?.message ??
                g?.podError}
            </span>
          )}
        </div>

        <p className="text-xs leading-relaxed text-muted-foreground">
          <b>Terminate, don&apos;t stop.</b> A stopped pod stays pinned to its host machine while
          that machine&apos;s GPU is re-rented, and the restart then fails with &ldquo;not enough
          free GPUs on the host machine&rdquo;. Billing ends either way, and the checkpoint lives on
          the network volume, so a fresh pod comes up with its weights already in place.
        </p>
      </Card>

      {/* ---- Machine --------------------------------------------------------- */}
      <Card className="space-y-4 p-6">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <HardDrive className="size-4 text-primary" />
            <h2 className="font-display text-lg font-medium">Machine</h2>
          </div>
          {rate !== null && (
            <span className="text-sm tabular-nums text-muted-foreground">
              ${rate.toFixed(2)}/hr
            </span>
          )}
        </div>

        {cat?.configured === false ? (
          <p className="text-sm text-muted-foreground">
            Add a RunPod API key below and this fills with the live catalog — every GPU that is
            purchasable right now, with its price.
          </p>
        ) : cat?.error ? (
          <p className="text-sm text-destructive">{cat.error}</p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            <Field
              label="GPU"
              hint={
                chosenGpu
                  ? `${chosenGpu.memory} GB · up to ${cloud === "SECURE" ? chosenGpu.maxCount.secure : chosenGpu.maxCount.community} available`
                  : "Live from RunPod's catalog, cheapest first."
              }
            >
              <select
                className={FIELD_CLASS}
                value={val("gpuTypeId", spec.gpuTypeId)}
                onChange={(e) => set("gpuTypeId", e.target.value)}
              >
                {gpus.length === 0 && <option value="">loading…</option>}
                {gpus.map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.name} · {x.memory}GB · ${priceOn(x).toFixed(2)}/hr
                  </option>
                ))}
              </select>
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Cloud" hint="Dedicated either way.">
                <select
                  className={FIELD_CLASS}
                  value={cloud}
                  onChange={(e) => set("gpuCloudType", e.target.value)}
                >
                  <option value="SECURE">Secure</option>
                  <option value="COMMUNITY">Community</option>
                </select>
              </Field>
              <Field label="Count" hint="1 is right here.">
                <select
                  className={FIELD_CLASS}
                  value={val("gpuCount", spec.gpuCount)}
                  onChange={(e) => set("gpuCount", e.target.value)}
                >
                  {[1, 2, 4, 8].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </Field>
            </div>

            <Field
              label="Region"
              hint="Volume-capable regions only. A volume is pinned to its region for life."
            >
              <select
                className={FIELD_CLASS}
                value={dcId}
                onChange={(e) => set("gpuDataCenterId", e.target.value)}
              >
                {(cat?.dataCenters ?? []).map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.id} · {d.region.toLowerCase().replace(/_/gu, " ")}
                  </option>
                ))}
              </select>
            </Field>

            <Field
              label="Network volume"
              hint="Holds the RF-DETR checkpoint and the video scratch. Survives every pod."
            >
              <select
                className={FIELD_CLASS}
                value={volumeId}
                onChange={(e) => set("gpuNetworkVolumeId", e.target.value)}
              >
                <option value="">none in {dcId}</option>
                {volumesHere.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} · {v.size}GB · {v.id}
                  </option>
                ))}
              </select>
            </Field>

            <div className="flex items-end gap-2 sm:col-span-2">
              <input
                className={FIELD_CLASS}
                value={volName}
                onChange={(e) => setVolName(e.target.value)}
                placeholder="new volume name"
              />
              <input
                className={`${FIELD_CLASS} w-24`}
                value={volSize}
                onChange={(e) => setVolSize(e.target.value)}
                placeholder="GB"
              />
              <Button
                size="sm"
                variant="outline"
                disabled={createVolume.isPending || !dcId}
                onClick={() => createVolume.mutate()}
              >
                {createVolume.isPending ? "Creating…" : "Create volume"}
              </Button>
            </div>
            {createVolume.error && (
              <span className="text-xs text-destructive sm:col-span-2">
                {createVolume.error.message}
              </span>
            )}
            <p className="text-xs leading-relaxed text-muted-foreground sm:col-span-2">
              A volume costs ~$0.07/GB/month and keeps billing when no pod exists — that is the
              price of not re-uploading a 260 MB checkpoint every time.
            </p>
          </div>
        )}
      </Card>

      {/* ---- Image and container env ---------------------------------------- */}
      <Card className="space-y-4 p-6">
        <div className="flex items-center gap-2">
          <Server className="size-4 text-primary" />
          <h2 className="font-display text-lg font-medium">Image and environment</h2>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Pod name" hint="Shown in the RunPod console.">
            <input
              className={FIELD_CLASS}
              value={val("gpuPodName", spec.name)}
              onChange={(e) => set("gpuPodName", e.target.value)}
            />
          </Field>
          <Field
            label="Image"
            hint={
              imgInfo
                ? `${imgInfo.isMlService ? "RF-DETR ML service" : "NOT the ML service"} · built ${imgInfo.createdAt?.slice(0, 10) ?? "?"} · ${imgInfo.digest?.slice(7, 19) ?? "?"}`
                : img?.status === "not-found"
                  ? "No such tag in the registry."
                  : imageCheck.isLoading
                    ? "Reading the registry…"
                    : "Could not verify — the registry did not answer."
            }
          >
            <input
              className={FIELD_CLASS}
              value={val("gpuImage", spec.image)}
              onChange={(e) => set("gpuImage", e.target.value)}
            />
          </Field>
          <Field
            label="Allowed CUDA versions"
            hint="Comma-separated. torch 2.12.1 is a cu13 build needing driver r580+; without this pin the pod can land on a 12.4 host and run on CPU at ~20x the wall-clock."
          >
            <input
              className={FIELD_CLASS}
              value={val("gpuCudaVersions", spec.allowedCudaVersions.join(","))}
              onChange={(e) => set("gpuCudaVersions", e.target.value)}
            />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Container disk (GB)" hint="Image layers only.">
              <input
                type="number"
                className={FIELD_CLASS}
                value={val("gpuContainerDiskGb", spec.containerDiskInGb)}
                onChange={(e) => set("gpuContainerDiskGb", e.target.value)}
              />
            </Field>
            <Field label="Volume mount" hint="Not the workdir.">
              <input
                className={FIELD_CLASS}
                value={val("gpuVolumeMountPath", spec.volumeMountPath)}
                onChange={(e) => set("gpuVolumeMountPath", e.target.value)}
              />
            </Field>
          </div>

          <Field label="Checkpoint path" hint="On the volume — it is not baked into the image.">
            <input
              className={FIELD_CLASS}
              value={val("mlWeightsPath", spec.env.RFDETR_WEIGHTS)}
              onChange={(e) => set("mlWeightsPath", e.target.value)}
            />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Batch" hint="Main GPU lever; also the fp16 trace shape.">
              <input
                type="number"
                className={FIELD_CLASS}
                value={val("mlBatch", spec.env.RFDETR_BATCH)}
                onChange={(e) => set("mlBatch", e.target.value)}
              />
            </Field>
            <Field label="Resolution" hint="What the checkpoint was trained at.">
              <input
                type="number"
                className={FIELD_CLASS}
                value={val("mlResolution", spec.env.RFDETR_RESOLUTION)}
                onChange={(e) => set("mlResolution", e.target.value)}
              />
            </Field>
          </div>

          <Field
            label="Media URL allowlist"
            hint="The SSRF gate: only these hosts may be fetched. Set it to the object-store host, or /analyze refuses every video."
          >
            <input
              className={FIELD_CLASS}
              value={val("mlMediaAllowlist", spec.env.MEDIA_URL_ALLOWLIST)}
              onChange={(e) => set("mlMediaAllowlist", e.target.value)}
              placeholder="media.example.com"
            />
          </Field>
          <Field
            label="Pod DATABASE_URL"
            hint="The pod writes detection rows directly. Reachable from RunPod, so not localhost."
          >
            <input
              type="password"
              className={FIELD_CLASS}
              value={draft.mlDatabaseUrl ?? ""}
              onChange={(e) => set("mlDatabaseUrl", e.target.value)}
              placeholder={s.mlDatabaseUrlMasked ?? "postgres://user:pass@host:5432/classroom"}
              autoComplete="off"
            />
          </Field>

          <Field
            label="SSH public key"
            hint="How the checkpoint gets onto a fresh volume — the image runs sshd for exactly this. Paste ~/.ssh/id_ed25519.pub."
            className="sm:col-span-2"
          >
            <textarea
              rows={2}
              className={`${FIELD_CLASS} font-mono text-xs`}
              value={draft.gpuSshPublicKey ?? ""}
              onChange={(e) => set("gpuSshPublicKey", e.target.value)}
              placeholder={
                s.sshPublicKeySet ? "•••••• saved — paste to replace" : "ssh-ed25519 AAAA… you@mac"
              }
            />
          </Field>

          <Check
            label="Spot (interruptible)"
            hint="Cheaper, but RunPod can reclaim the pod mid-lesson. The job requeues; the GPU-minutes spent do not come back."
            checked={flag("gpuInterruptible", spec.interruptible)}
            onChange={(v) => set("gpuInterruptible", String(v))}
          />
          <Check
            label="TensorRT backend"
            hint="Off until tools/trt_parity.py has proven, on this GPU, that the engine agrees with PyTorch and is actually faster than fp16."
            checked={flag("mlTensorrt", spec.env.RFDETR_TENSORRT === "true")}
            onChange={(v) => set("mlTensorrt", String(v))}
          />
        </div>
      </Card>

      {/* ---- Autopilot and connection ---------------------------------------- */}
      <Card className="space-y-4 p-6">
        <div className="flex items-center gap-2">
          <Zap className="size-4 text-primary" />
          <h2 className="font-display text-lg font-medium">Autopilot and connection</h2>
        </div>
        <form
          className="grid gap-4 sm:grid-cols-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (dirty) save.mutate();
          }}
        >
          <Check
            label="Auto-provision GPU"
            hint="Queued lesson and no GPU serving it: create a pod from this spec, or start a stopped one. This is what makes an empty account recover without anyone opening a browser."
            checked={flag("gpuAutoStart", s.gpuAutoStart)}
            onChange={(v) => set("gpuAutoStart", String(v))}
          />
          <div className="grid grid-cols-2 gap-3">
            <Field label="Idle release (min)" hint="0 disables.">
              <input
                type="number"
                min={0}
                max={1440}
                className={FIELD_CLASS}
                value={val("gpuAutoStopMinutes", s.gpuAutoStopMinutes)}
                onChange={(e) => set("gpuAutoStopMinutes", e.target.value)}
              />
            </Field>
            <Field label="When idle" hint="Terminate is safer.">
              <select
                className={FIELD_CLASS}
                value={val("gpuIdleAction", s.gpuIdleAction)}
                onChange={(e) => set("gpuIdleAction", e.target.value)}
              >
                <option value="terminate">Terminate</option>
                <option value="stop">Stop</option>
              </select>
            </Field>
          </div>

          <Field
            label="RunPod API key"
            hint="Stored server-side, shown masked. Leave blank to keep the current key."
          >
            <input
              type="password"
              className={FIELD_CLASS}
              value={draft.runpodApiKey ?? ""}
              onChange={(e) => set("runpodApiKey", e.target.value)}
              placeholder={s.runpodApiKeyMasked ?? "rpa_…"}
              autoComplete="off"
            />
          </Field>
          <Field
            label="Pod ID"
            hint="Written when this app creates a pod, cleared when it terminates one. Type an id here only to adopt a pod created elsewhere."
          >
            <input
              className={FIELD_CLASS}
              value={val("runpodPodId", s.runpodPodId)}
              onChange={(e) => set("runpodPodId", e.target.value)}
              placeholder="none"
            />
          </Field>

          <Field
            label="ML service URL"
            hint="Leave empty. With a pod ID set, the URL is derived from it — RunPod's proxy hostname is the one address that survives a pod's whole life, so autopilot never leaves a stale URL behind. Fill it in only to override (a tunnel, a second pod)."
            className="sm:col-span-2"
          >
            <input
              className={FIELD_CLASS}
              value={val("mlServiceUrl", s.mlServiceUrl)}
              onChange={(e) => set("mlServiceUrl", e.target.value)}
              placeholder={s.mlServiceUrlEffective ?? s.mlServiceUrlDefault}
            />
          </Field>

          <div className="flex items-center gap-3 sm:col-span-2">
            <Button type="submit" size="sm" disabled={!dirty || save.isPending}>
              {save.isPending ? "Saving…" : "Save settings"}
            </Button>
            {dirty && (
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => {
                  setDraft({});
                  setSaved(false);
                }}
              >
                Discard
              </Button>
            )}
            {saved && !dirty && <span className="text-xs text-muted-foreground">Saved.</span>}
            {save.isError && <span className="text-xs text-destructive">{save.error.message}</span>}
          </div>
        </form>
      </Card>
    </div>
  );
}

/** `{key: value}` when the field was edited, `{}` when it was not. */
function def<K extends string, V>(
  key: K,
  value: V | undefined,
): Record<K, V> | Record<never, never> {
  return value === undefined ? {} : ({ [key]: value } as Record<K, V>);
}
