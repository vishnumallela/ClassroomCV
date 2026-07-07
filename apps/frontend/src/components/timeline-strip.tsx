import { Card } from "@/components/ui/card";

type Interval = [number, number];

function Lane({
  label,
  intervals,
  color,
  durationMs,
}: {
  label: string;
  intervals: Interval[];
  color: string;
  durationMs: number;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-xs text-muted-foreground">{label}</span>
      <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-muted">
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
    </div>
  );
}

export function TimelineStrip({
  durationMs,
  presenceIntervals,
  boardIntervals,
  currentMs,
  onSeek,
}: {
  durationMs: number | null;
  presenceIntervals: Interval[];
  boardIntervals: Interval[];
  currentMs: number;
  onSeek: (ms: number) => void;
}) {
  if (!durationMs || durationMs <= 0) return null;
  const playheadPct = Math.max(0, Math.min(1, currentMs / durationMs)) * 100;

  return (
    <Card className="p-4">
      <div className="mb-3 text-sm font-medium text-muted-foreground">Timeline</div>
      <div className="relative">
        <button
          type="button"
          aria-label="Seek"
          className="absolute inset-0 z-10 cursor-pointer"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            onSeek(((e.clientX - rect.left) / rect.width) * durationMs);
          }}
        />
        <div className="space-y-2">
          <Lane
            label="Teacher present"
            intervals={presenceIntervals}
            color="bg-primary"
            durationMs={durationMs}
          />
          <Lane
            label="At board"
            intervals={boardIntervals}
            color="bg-amber-400"
            durationMs={durationMs}
          />
        </div>
        <div
          className="pointer-events-none absolute inset-y-0 z-20 w-px bg-foreground"
          style={{ left: `calc(6rem + 0.75rem + (100% - 6rem - 0.75rem) * ${playheadPct / 100})` }}
        />
      </div>
    </Card>
  );
}
