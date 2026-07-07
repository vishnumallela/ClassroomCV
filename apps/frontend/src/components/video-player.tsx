import type * as React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  colorFor,
  countRoles,
  type DetectionData,
  fetchDetections,
  findFrameIndex,
  ROLE_COLORS,
  roleLabel,
} from "@/components/detections";
import { cn } from "@/lib/utils";

type Mode = "raw" | "detections";
type LoadState = "idle" | "loading" | "ready" | "empty" | "error";

interface PlayerZone {
  kind: string;
  polygon: [number, number][];
}

interface VideoPlayerProps {
  videoId: string;
  streamUrl: string;
  width: number | null;
  height: number | null;
  analyticsReady: boolean;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  zones?: PlayerZone[];
  onTimeUpdate?: (ms: number) => void;
}

const ZONE_COLORS: Record<string, { stroke: string; fill: string; label: string }> = {
  board: { stroke: "#facc15", fill: "rgba(250,204,21,0.10)", label: "Board" },
  door: { stroke: "#34d399", fill: "rgba(52,211,153,0.10)", label: "Door" },
};

const LABEL_FONT = '600 12px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';

export function VideoPlayer({
  videoId,
  streamUrl,
  width,
  height,
  analyticsReady,
  videoRef,
  zones,
  onTimeUpdate,
}: VideoPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<DetectionData | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [mode, setMode] = useState<Mode>("raw");

  // Fetch overlay once analysis is done; default to Detections when boxes exist.
  useEffect(() => {
    if (!analyticsReady) {
      setLoadState("idle");
      setData(null);
      setMode("raw");
      return;
    }
    let cancelled = false;
    setLoadState("loading");
    void (async () => {
      try {
        const d = await fetchDetections(videoId);
        if (cancelled) return;
        setData(d);
        if (d.frames.length > 0) {
          setLoadState("ready");
          setMode("detections");
        } else {
          setLoadState("empty");
          setMode("raw");
        }
      } catch {
        if (!cancelled) {
          setLoadState("error");
          setMode("raw");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [analyticsReady, videoId]);

  const showOverlay = mode === "detections" && loadState === "ready";

  const draw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cw = video.clientWidth;
    const ch = video.clientHeight;
    if (cw === 0 || ch === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(cw * dpr);
    const bh = Math.round(ch * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (!showOverlay || !data) return;

    // The <video> is object-contain letterboxed; map normalized boxes into that rect.
    const vw = video.videoWidth || data.width || width || cw;
    const vh = video.videoHeight || data.height || height || ch;
    const scale = Math.min(cw / vw, ch / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    const ox = (cw - dw) / 2;
    const oy = (ch - dh) / 2;

    ctx.font = LABEL_FONT;

    for (const zone of zones ?? []) {
      const style = ZONE_COLORS[zone.kind];
      if (!style || zone.polygon.length < 3) continue;
      ctx.beginPath();
      zone.polygon.forEach(([zx, zy], i) => {
        const px = ox + zx * dw;
        const py = oy + zy * dh;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.globalAlpha = 1;
      ctx.fillStyle = style.fill;
      ctx.fill();
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = style.stroke;
      ctx.stroke();
      ctx.setLineDash([]);
      const zx0 = ox + Math.min(...zone.polygon.map((p) => p[0])) * dw;
      const zy0 = oy + Math.min(...zone.polygon.map((p) => p[1])) * dh;
      ctx.fillStyle = style.stroke;
      ctx.fillRect(zx0, Math.max(oy, zy0 - 15), ctx.measureText(style.label).width + 10, 15);
      ctx.fillStyle = "#000000";
      ctx.fillText(style.label, zx0 + 5, Math.max(oy, zy0 - 15) + 11);
    }

    const frameIdx = findFrameIndex(data.frames, video.currentTime * 1000);
    if (frameIdx < 0) return;

    for (const [trackNo, x, y, w, h] of data.frames[frameIdx]!.boxes) {
      const role = data.roles[String(trackNo)] ?? "unknown";
      const isTeacher = role === "teacher";
      const color = colorFor(role);
      const px = ox + x * dw;
      const py = oy + y * dh;
      const pw = w * dw;
      const ph = h * dh;

      ctx.globalAlpha = role === "unknown" ? 0.7 : 1;
      ctx.lineWidth = isTeacher ? 3 : 1.75;
      ctx.strokeStyle = color;
      ctx.strokeRect(px, py, pw, ph);

      const label = `${roleLabel(role)} ${trackNo}`;
      const padX = 5;
      const labelH = 15;
      const tw = ctx.measureText(label).width + padX * 2;
      let ly = py - labelH;
      if (ly < oy) ly = Math.min(py + ph, ch - labelH);
      let lx = px;
      if (lx + tw > cw) lx = Math.max(0, cw - tw);

      ctx.globalAlpha = 1;
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly, tw, labelH);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, lx + padX, ly + 11);
    }
  }, [showOverlay, data, width, height, zones, videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let raf = 0;
    const loop = () => {
      draw();
      raf = requestAnimationFrame(loop);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      loop();
    };
    const stop = () => {
      cancelAnimationFrame(raf);
      draw();
    };
    const once = () => draw();

    video.addEventListener("play", start);
    video.addEventListener("playing", start);
    video.addEventListener("pause", stop);
    video.addEventListener("ended", stop);
    video.addEventListener("seeked", once);
    video.addEventListener("timeupdate", once);
    video.addEventListener("loadedmetadata", once);
    window.addEventListener("resize", once);
    document.addEventListener("fullscreenchange", once);
    const ro = new ResizeObserver(() => draw());
    ro.observe(video);

    draw();
    if (!video.paused && !video.ended) loop();

    return () => {
      cancelAnimationFrame(raf);
      video.removeEventListener("play", start);
      video.removeEventListener("playing", start);
      video.removeEventListener("pause", stop);
      video.removeEventListener("ended", stop);
      video.removeEventListener("seeked", once);
      video.removeEventListener("timeupdate", once);
      video.removeEventListener("loadedmetadata", once);
      window.removeEventListener("resize", once);
      document.removeEventListener("fullscreenchange", once);
      ro.disconnect();
    };
  }, [draw, videoRef]);

  const counts = data ? countRoles(data.roles) : null;

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-xl border border-border bg-black">
        <video
          ref={videoRef}
          src={streamUrl}
          controls
          className="block aspect-video w-full bg-black"
          onTimeUpdate={(e) => onTimeUpdate?.(e.currentTarget.currentTime * 1000)}
        >
          <track kind="captions" />
        </video>
        <canvas
          ref={canvasRef}
          className={cn(
            "pointer-events-none absolute inset-0 h-full w-full",
            !showOverlay && "hidden",
          )}
        />
      </div>

      {loadState === "ready" && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["detections", "raw"] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  "rounded px-3 py-1 font-medium transition-colors",
                  mode === m
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {m === "detections" ? "Detections" : "Raw"}
              </button>
            ))}
          </div>
          {counts && (
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5">
                <span
                  className="size-2.5 rounded-full"
                  style={{ backgroundColor: ROLE_COLORS.teacher }}
                />
                {counts.teacher} teacher
              </span>
              <span className="flex items-center gap-1.5">
                <span
                  className="size-2.5 rounded-full"
                  style={{ backgroundColor: ROLE_COLORS.student }}
                />
                {counts.student} students
              </span>
            </div>
          )}
        </div>
      )}
      {loadState === "empty" && (
        <p className="text-xs text-muted-foreground">
          No detections were produced for this recording.
        </p>
      )}
    </div>
  );
}
