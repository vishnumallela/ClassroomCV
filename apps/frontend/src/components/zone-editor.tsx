import { useQueryClient } from "@tanstack/react-query";
import type { RouterOutputs } from "@classroom/api-contracts";
import { type MouseEvent, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { orpcClient } from "@/lib/orpc";
import { cn } from "@/lib/utils";

type ZoneKind = "board" | "door";
type Point = [number, number];
type Zone = RouterOutputs["videos"]["get"]["zones"][number];
type DraftZone = { kind: ZoneKind; polygon: Point[] };

const ZONE_STYLE: Record<ZoneKind, { label: string; stroke: string; fill: string }> = {
  board: { label: "Board", stroke: "#facc15", fill: "rgba(250,204,21,0.16)" },
  door: { label: "Door", stroke: "#34d399", fill: "rgba(52,211,153,0.16)" },
};

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
const toSvg = (pts: Point[]) => pts.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");
const isKind = (k: string): k is ZoneKind => k === "board" || k === "door";

export function ZoneEditor({
  videoId,
  frameSrc,
  aspect,
  initialZones,
  onClose,
}: {
  videoId: string;
  frameSrc: string | null;
  aspect: number;
  initialZones: Zone[];
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const stageRef = useRef<HTMLButtonElement>(null);
  const [zones, setZones] = useState<DraftZone[]>(
    initialZones
      .filter((z) => isKind(z.kind))
      .map((z) => ({ kind: z.kind as ZoneKind, polygon: z.polygon as Point[] })),
  );
  const [draft, setDraft] = useState<Point[]>([]);
  const [activeKind, setActiveKind] = useState<ZoneKind>("board");
  const [cursor, setCursor] = useState<Point | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const pointFromClient = (clientX: number, clientY: number): Point => {
    const rect = stageRef.current!.getBoundingClientRect();
    return [
      clamp01((clientX - rect.left) / rect.width),
      clamp01((clientY - rect.top) / rect.height),
    ];
  };

  const closeDraft = () => {
    setDraft((d) => {
      if (d.length < 3) return d;
      setZones((zs) => [
        ...zs.filter((z) => z.kind !== activeKind),
        { kind: activeKind, polygon: d },
      ]);
      return [];
    });
    setNote(null);
  };

  const latest = useRef({ closeDraft, onClose, hasDraft: draft.length > 0 });
  latest.current = { closeDraft, onClose, hasDraft: draft.length > 0 };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (latest.current.hasDraft) setDraft([]);
        else latest.current.onClose();
      } else if (e.key === "Enter") {
        latest.current.closeDraft();
      } else if (e.key === "Backspace") {
        setDraft((d) => d.slice(0, -1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const detectBoard = async () => {
    setDetecting(true);
    setNote(null);
    try {
      const res = await orpcClient.board.detect({ id: videoId });
      if (res.polygon && res.polygon.length >= 3) {
        const poly = res.polygon.map(([x, y]) => [clamp01(x), clamp01(y)] as Point);
        setZones((zs) => [
          ...zs.filter((z) => z.kind !== "board"),
          { kind: "board", polygon: poly },
        ]);
        setActiveKind("board");
        setNote(`Board detected at ${Math.round(res.confidence * 100)}% confidence.`);
      } else {
        setNote("No board found automatically. Draw it by hand.");
      }
    } catch {
      setNote("Board detection failed.");
    } finally {
      setDetecting(false);
    }
  };

  const save = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await orpcClient.zones.upsert({
        id: videoId,
        zones: zones.map((z) => ({ kind: z.kind, polygon: z.polygon })),
      });
      await queryClient.invalidateQueries();
      onClose();
    } catch {
      setSaveError("Save failed. The recording may still be processing.");
      setSaving(false);
    }
  };

  const last = draft.at(-1);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-3xl rounded-xl border border-border bg-card p-4 shadow-xl">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold tracking-tight">Edit zones</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>

        <button
          ref={stageRef}
          type="button"
          className="relative block w-full cursor-crosshair overflow-hidden rounded-lg border border-border bg-black"
          style={{ aspectRatio: String(aspect) }}
          onClick={(e: MouseEvent) =>
            setDraft((d) => [...d, pointFromClient(e.clientX, e.clientY)])
          }
          onMouseMove={(e: MouseEvent) => setCursor(pointFromClient(e.clientX, e.clientY))}
          onDoubleClick={closeDraft}
        >
          {frameSrc && (
            <img
              src={frameSrc}
              alt=""
              className="absolute inset-0 h-full w-full object-fill opacity-80"
            />
          )}
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="absolute inset-0 h-full w-full"
          >
            {zones.map((z) => (
              <polygon
                key={z.kind}
                points={toSvg(z.polygon)}
                stroke={ZONE_STYLE[z.kind].stroke}
                fill={ZONE_STYLE[z.kind].fill}
                strokeWidth={0.6}
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {draft.length > 0 && (
              <polyline
                points={toSvg(draft)}
                stroke={ZONE_STYLE[activeKind].stroke}
                fill="none"
                strokeWidth={0.6}
                vectorEffect="non-scaling-stroke"
              />
            )}
            {last && cursor && (
              <line
                x1={last[0] * 100}
                y1={last[1] * 100}
                x2={cursor[0] * 100}
                y2={cursor[1] * 100}
                stroke={ZONE_STYLE[activeKind].stroke}
                strokeDasharray="1 1"
                strokeWidth={0.5}
                vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>
        </button>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["board", "door"] as ZoneKind[]).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setActiveKind(k)}
                className={cn(
                  "rounded px-3 py-1 font-medium",
                  activeKind === k
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {ZONE_STYLE[k].label}
              </button>
            ))}
          </div>
          <Button variant="outline" size="sm" disabled={detecting} onClick={detectBoard}>
            {detecting ? "Detecting" : "Auto-detect board"}
          </Button>
          {zones.map((z) => (
            <Button
              key={z.kind}
              variant="ghost"
              size="sm"
              onClick={() => setZones((zs) => zs.filter((x) => x.kind !== z.kind))}
            >
              Clear {ZONE_STYLE[z.kind].label.toLowerCase()}
            </Button>
          ))}
          <Button
            className="ml-auto"
            size="sm"
            disabled={saving || draft.length > 0}
            onClick={save}
          >
            {saving ? "Saving" : "Save zones"}
          </Button>
        </div>

        <p className="mt-2 text-xs text-muted-foreground">
          {draft.length > 0
            ? "Click to add points, then double-click or press Enter to close the shape."
            : (note ?? "Click on the frame to draw a zone, or auto-detect the board.")}
        </p>
        {saveError && <p className="mt-1 text-xs text-destructive">{saveError}</p>}
      </div>
    </div>
  );
}
