import type { RouterOutputs } from "@classroom/api-contracts";
import { useEffect, useMemo, useRef } from "react";
import { Card } from "@/components/ui/card";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;

// Perceptual low->high ramp (deep blue -> cyan -> green -> amber -> red).
const RAMP: [number, number, number][] = [
  [13, 71, 161],
  [0, 188, 212],
  [76, 175, 80],
  [255, 193, 7],
  [244, 67, 54],
];

function sampleRamp(t: number): [number, number, number] {
  const n = RAMP.length;
  const x = Math.max(0, Math.min(1, t)) * (n - 1);
  const i = Math.min(n - 1, Math.floor(x));
  const f = x - i;
  const a = RAMP[i] as [number, number, number];
  const b = RAMP[Math.min(n - 1, i + 1)] as [number, number, number];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

const RENDER_W = 640;

/** Teacher movement heatmap — one of the three KPIs. Teacher-only since the
 * 2026-08 slimming (the students grid is no longer computed or stored). */
export function HeatmapCard({
  analytics,
  thumbnailUrl,
  aspect,
}: {
  analytics: Analytics;
  thumbnailUrl: string | null;
  aspect: number;
}) {
  const hm = analytics.heatmap;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const total = useMemo(() => hm?.teacher?.reduce((s, v) => s + v, 0) ?? 0, [hm]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !hm || hm.grid_w === 0) return;
    const cells = hm.teacher;
    const max = cells.reduce((m, v) => Math.max(m, v), 0);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (max === 0) return;

    // Paint at grid resolution, then upscale with smoothing for a soft heatmap.
    const small = document.createElement("canvas");
    small.width = hm.grid_w;
    small.height = hm.grid_h;
    const sctx = small.getContext("2d");
    if (!sctx) return;
    const img = sctx.createImageData(hm.grid_w, hm.grid_h);
    for (let i = 0; i < cells.length; i++) {
      // sqrt lifts sparsely-visited cells so a dominant spot (the board) does
      // not wash the rest of the path out.
      const norm = Math.sqrt((cells[i] ?? 0) / max);
      const [r, g, b] = sampleRamp(norm);
      const o = i * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = Math.round(Math.min(0.85, norm) * 255);
    }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
  }, [hm]);

  if (!hm || hm.grid_w === 0 || total === 0) {
    return (
      <Card className="p-6 text-sm text-muted-foreground">
        No teacher movement to map yet.
      </Card>
    );
  }

  const renderH = Math.round(RENDER_W / (aspect || 16 / 9));

  return (
    <Card className="overflow-hidden">
      <div className="border-b border-border p-4">
        <h3 className="text-sm font-medium">Teacher movement heatmap</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Where the teacher spent the lesson: board, aisles, and the desks she visited.
        </p>
      </div>

      <div
        className="hud-corners relative w-full bg-muted"
        style={{ aspectRatio: String(aspect || 16 / 9) }}
      >
        {thumbnailUrl && (
          <img
            src={thumbnailUrl}
            alt=""
            crossOrigin="use-credentials"
            className="absolute inset-0 h-full w-full object-cover opacity-40"
          />
        )}
        <canvas
          ref={canvasRef}
          width={RENDER_W}
          height={renderH}
          className="absolute inset-0 h-full w-full"
        />
      </div>

      <div className="flex items-center gap-2 p-3 text-xs text-muted-foreground">
        <span>Less time</span>
        <span
          className="h-2 flex-1 rounded-full"
          style={{
            background:
              "linear-gradient(90deg, rgb(13,71,161), rgb(0,188,212), rgb(76,175,80), rgb(255,193,7), rgb(244,67,54))",
          }}
        />
        <span>More time</span>
      </div>
    </Card>
  );
}
