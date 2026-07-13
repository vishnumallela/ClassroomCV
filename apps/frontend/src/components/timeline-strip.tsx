import type { RouterOutputs } from "@classroom/api-contracts";
import { KIND_LABEL } from "@/components/events-table";
import { Card } from "@/components/ui/card";
import { type TeacherState, teacherStateSegments } from "@/lib/analytics";
import { msToClock } from "@/lib/format";

type Interval = [number, number];
type VideoEvent = RouterOutputs["videos"]["get"]["events"][number];

const STATE_COLOR: Record<TeacherState, string> = {
  absent: "bg-zinc-300 dark:bg-zinc-700",
  circulating: "bg-primary",
  board: "bg-amber-400",
};
const STATE_LABEL: Record<TeacherState, string> = {
  absent: "Out of room",
  circulating: "Circulating",
  board: "At board",
};

function StateLane({
  presenceIntervals,
  boardIntervals,
  durationMs,
  onSeek,
}: {
  presenceIntervals: Interval[];
  boardIntervals: Interval[];
  durationMs: number;
  onSeek: (ms: number) => void;
}) {
  const segments = teacherStateSegments(presenceIntervals, boardIntervals, durationMs);
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-xs text-muted-foreground">Teacher state</span>
      <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-zinc-300 dark:bg-zinc-700">
        {segments.map((s) => (
          <button
            key={`${s.state}-${s.start}`}
            type="button"
            title={`${STATE_LABEL[s.state]} · ${msToClock(s.start)}–${msToClock(s.end)}`}
            aria-label={`${STATE_LABEL[s.state]} at ${msToClock(s.start)}`}
            // z-20 beats the strip-wide seek overlay (z-10); without it the
            // overlay eats these segments' hover tooltip and snap-to-start seek.
            className={`absolute inset-y-0 z-20 ${STATE_COLOR[s.state]}`}
            style={{
              left: `${(s.start / durationMs) * 100}%`,
              width: `${Math.max(0.3, ((s.end - s.start) / durationMs) * 100)}%`,
            }}
            onClick={() => onSeek(s.start)}
          />
        ))}
      </div>
    </div>
  );
}

const TICK_COLOR: Record<string, string> = {
  enter: "bg-emerald-500",
  exit: "bg-red-500",
  board_enter: "bg-amber-600",
  board_leave: "bg-amber-600",
};

// Lane bars start after the w-24 (6rem) label column and its gap-3 (0.75rem);
// every strip-wide absolutely positioned element (seek overlay, playhead) must
// skip the same offset or its x-axis drifts away from the lane segments.
const LANE_OFFSET = "6rem + 0.75rem";

function Lane({
  label,
  intervals,
  color,
  durationMs,
  ticks,
  onSeek,
}: {
  label: string;
  intervals: Interval[];
  color: string;
  durationMs: number;
  ticks: VideoEvent[];
  onSeek: (ms: number) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-xs text-muted-foreground">{label}</span>
      {/* Ticks live outside the overflow-hidden bar so they can extend past it. */}
      <div className="relative h-3 flex-1">
        <div className="absolute inset-0 overflow-hidden rounded-full bg-muted">
          {intervals.map((iv) => (
            <div
              key={`${iv[0]}-${iv[1]}`}
              className={`absolute inset-y-0 rounded-full ${color}`}
              style={{
                left: `${(iv[0] / durationMs) * 100}%`,
                width: `${Math.max(0.4, ((iv[1] - iv[0]) / durationMs) * 100)}%`,
              }}
            />
          ))}
        </div>
        {ticks.map((e) => {
          const title = `${KIND_LABEL[e.kind] ?? e.kind} at ${msToClock(e.videoTsMs)}`;
          return (
            <button
              key={`${e.kind}-${e.videoTsMs}-${e.trackNo}`}
              type="button"
              title={title}
              aria-label={title}
              // z-20 beats the strip-wide seek overlay (z-10); the button is
              // wider than the 2px line purely as a click target, with the
              // visible line centered on the exact videoTsMs position.
              className="absolute -inset-y-1 z-20 w-2 -translate-x-1/2 cursor-pointer"
              style={{ left: `${(e.videoTsMs / durationMs) * 100}%` }}
              onClick={() => onSeek(e.videoTsMs)}
            >
              <span
                className={`absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 rounded-full ${
                  TICK_COLOR[e.kind] ?? "bg-foreground"
                }`}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function TimelineStrip({
  durationMs,
  presenceIntervals,
  boardIntervals,
  events,
  currentMs,
  onSeek,
}: {
  durationMs: number | null;
  presenceIntervals: Interval[];
  boardIntervals: Interval[];
  events: VideoEvent[];
  currentMs: number;
  onSeek: (ms: number) => void;
}) {
  if (!durationMs || durationMs <= 0) return null;
  const playheadPct = Math.max(0, Math.min(1, currentMs / durationMs)) * 100;
  const roomTicks = events.filter((e) => e.kind === "enter" || e.kind === "exit");
  const boardTicks = events.filter((e) => e.kind === "board_enter" || e.kind === "board_leave");

  return (
    <Card className="p-4">
      <div className="mb-3 text-sm font-medium text-muted-foreground">Timeline</div>
      <div className="relative">
        <button
          type="button"
          aria-label="Seek"
          className="absolute inset-y-0 right-0 z-10 cursor-pointer"
          style={{ left: `calc(${LANE_OFFSET})` }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            onSeek(((e.clientX - rect.left) / rect.width) * durationMs);
          }}
        />
        <div className="space-y-2">
          <StateLane
            presenceIntervals={presenceIntervals}
            boardIntervals={boardIntervals}
            durationMs={durationMs}
            onSeek={onSeek}
          />
          <Lane
            label="Teacher present"
            intervals={presenceIntervals}
            color="bg-primary"
            durationMs={durationMs}
            ticks={roomTicks}
            onSeek={onSeek}
          />
          <Lane
            label="At board"
            intervals={boardIntervals}
            color="bg-amber-400"
            durationMs={durationMs}
            ticks={boardTicks}
            onSeek={onSeek}
          />
        </div>
        <div
          className="pointer-events-none absolute inset-y-0 z-20 w-px bg-foreground"
          style={{
            left: `calc(${LANE_OFFSET} + (100% - (${LANE_OFFSET})) * ${playheadPct / 100})`,
          }}
        />
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 pl-[calc(6rem+0.75rem)] text-xs text-muted-foreground">
        {(["board", "circulating", "absent"] as const).map((s) => (
          <span key={s} className="flex items-center gap-1.5">
            <span className={`size-2 rounded-full ${STATE_COLOR[s]}`} />
            {STATE_LABEL[s]}
          </span>
        ))}
      </div>
    </Card>
  );
}
