import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";
import { Card } from "@/components/ui/card";
import { orpc } from "@/lib/orpc";

// Clean empty-room plate (student desks removed) as the floor backdrop. It was
// generated to match the fixed camera's framing, so teacher foot points in
// video-normalized coords fall on the room floor with no per-video transform.
// (Do NOT feature-warp this plate to a video frame — SIFT matches on the wall
// posters/board, which drags the floor plane up and lifts the heat onto the wall.)
// ?v=2 busts the browser cache from the earlier (feature-warped) plate file.
const ROOM_PLATE = "/room-plate.png?v=2";
const W = 960;

// Dwell palette: transparent -> blue -> teal -> green -> yellow -> orange -> red,
// with alpha rising so lightly-visited floor is faint and hotspots are solid.
const PALETTE_STOPS: [number, string][] = [
  [0.0, "rgba(0,0,255,0)"],
  [0.1, "rgba(0,150,255,0.25)"],
  [0.3, "rgba(0,210,150,0.5)"],
  [0.5, "rgba(130,220,60,0.62)"],
  [0.7, "rgba(240,230,40,0.72)"],
  [0.86, "rgba(250,140,20,0.8)"],
  [1.0, "rgba(230,20,20,0.85)"],
];

function buildPalette(): Uint8ClampedArray {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 1;
  const ctx = c.getContext("2d")!;
  const g = ctx.createLinearGradient(0, 0, 256, 0);
  for (const [stop, color] of PALETTE_STOPS) g.addColorStop(stop, color);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 1);
  return ctx.getImageData(0, 0, 256, 1).data;
}

export function HeatmapCard({ videoId, aspect }: { videoId: string; aspect: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { data } = useQuery(
    orpc.videos.detections.queryOptions({ input: { id: videoId, fps: 5 } }),
  );

  // Teacher foot points (bbox bottom-centre = where she stands on the floor).
  // Each sampled frame contributes equal weight, so accumulated density == dwell.
  const feet = useMemo<[number, number][]>(() => {
    if (!data) return [];
    const teacherNo = Object.entries(data.roles).find(([, r]) => r === "teacher")?.[0];
    if (teacherNo === undefined) return [];
    const tn = Number(teacherNo);
    const pts: [number, number][] = [];
    for (const frame of data.frames) {
      const b = frame.boxes.find((x) => x[0] === tn);
      if (b) pts.push([b[1] + b[3] / 2, b[2] + b[4]]);
    }
    return pts;
  }, [data]);

  const aspectRatio = aspect > 0 ? aspect : 16 / 9;
  const H = Math.round(W / aspectRatio);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || feet.length === 0) return;
    ctx.clearRect(0, 0, W, H);

    // 1) Accumulate each foot point as a soft Gaussian kernel in FLOAT space.
    //    Canvas-alpha accumulation clips at 255: with hundreds of overlapping
    //    samples a wide area saturates and renders as one flat red blob. Float
    //    accumulation keeps the full dynamic range so dwell stays graded.
    const R = Math.round(W * 0.033);
    const sigma2 = (R / 2.5) ** 2;
    const side = 2 * R + 1;
    const kernel = new Float32Array(side * side);
    for (let dy = -R; dy <= R; dy++) {
      for (let dx = -R; dx <= R; dx++) {
        const d2 = dx * dx + dy * dy;
        kernel[(dy + R) * side + (dx + R)] =
          d2 <= R * R ? Math.exp(-d2 / (2 * sigma2)) : 0;
      }
    }
    const acc = new Float32Array(W * H);
    for (const [x, y] of feet) {
      const cx = Math.round(x * W);
      const cy = Math.round(y * H);
      for (let dy = -R; dy <= R; dy++) {
        const py = cy + dy;
        if (py < 0 || py >= H) continue;
        for (let dx = -R; dx <= R; dx++) {
          const px = cx + dx;
          if (px < 0 || px >= W) continue;
          acc[py * W + px]! += kernel[(dy + R) * side + (dx + R)]!;
        }
      }
    }

    // 2) Normalize to the busiest cell, then colorize through the palette —
    //    the hottest dwell spot maps to red, everything else grades down.
    let max = 0;
    for (let i = 0; i < acc.length; i++) if (acc[i]! > max) max = acc[i]!;
    if (max <= 0) return;
    const palette = buildPalette();
    const out = ctx.createImageData(W, H);
    for (let i = 0; i < acc.length; i++) {
      const p = Math.min(255, Math.round((acc[i]! / max) * 255)) * 4;
      out.data[i * 4] = palette[p]!;
      out.data[i * 4 + 1] = palette[p + 1]!;
      out.data[i * 4 + 2] = palette[p + 2]!;
      out.data[i * 4 + 3] = palette[p + 3]!;
    }
    ctx.putImageData(out, 0, 0);
  }, [feet, H]);

  return (
    <Card className="overflow-hidden">
      <div className="border-b border-border p-4">
        <h3 className="text-sm font-medium">Teacher floor heatmap</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Where the teacher spent the lesson on the floor — warmer areas mean more time.
        </p>
      </div>

      <div className="relative w-full bg-muted" style={{ aspectRatio: String(aspectRatio) }}>
        <img src={ROOM_PLATE} alt="" className="absolute inset-0 h-full w-full object-cover" />
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          className="absolute inset-0 h-full w-full"
        />
        {feet.length === 0 && (
          <div className="absolute inset-0 grid place-items-center bg-black/40 text-sm text-white">
            No teacher movement to map.
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 p-3 text-xs text-muted-foreground">
        <span>Less time</span>
        <span
          className="h-2 flex-1 rounded-full"
          style={{
            background:
              "linear-gradient(90deg, rgba(0,150,255,0.4), rgba(0,210,150,0.6), rgb(130,220,60), rgb(240,230,40), rgb(250,140,20), rgb(230,20,20))",
          }}
        />
        <span>More time</span>
      </div>
    </Card>
  );
}
