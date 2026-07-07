import type { RouterOutputs } from "@classroom/api-contracts";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import { peakOccupancy } from "@/lib/analytics";
import { msToClock } from "@/lib/format";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;

const EMERALD = "#10b981";
const AMBER = "#f59e0b";
const AXIS = "#71717a";
const GRID = "rgba(113,113,122,0.18)";

interface TooltipEntry {
  value?: number;
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string | number;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-lg border border-border bg-popover/85 px-3 py-2 text-xs shadow-lg backdrop-blur">
      <div className="font-medium tabular-nums">{msToClock(Number(label))}</div>
      <div className="mt-0.5 flex items-center gap-1.5 text-muted-foreground">
        <span className="size-2 rounded-full" style={{ backgroundColor: EMERALD }} />
        {payload[0]?.value ?? 0} students
      </div>
    </div>
  );
}

export function OccupancyChart({
  analytics,
  durationMs,
  onSeek,
}: {
  analytics: Analytics;
  durationMs: number | null;
  onSeek: (ms: number) => void;
}) {
  const data = analytics.occupancy.map((p) => ({ ts: p.ts_ms, students: p.students }));
  if (data.length === 0) return null;
  const domainMax = durationMs && durationMs > 0 ? durationMs : (data.at(-1)?.ts ?? 0);
  const peak = peakOccupancy(analytics.occupancy);

  return (
    <Card className="p-4">
      <div className="mb-3 text-sm font-medium text-muted-foreground">
        Student occupancy over time
      </div>
      <div className="relative h-60 w-full">
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage: "radial-gradient(rgba(113,113,122,0.22) 1px, transparent 1px)",
            backgroundSize: "16px 16px",
          }}
        />
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 6, right: 10, bottom: 0, left: 0 }}
            onClick={(state: { activeLabel?: string | number }) => {
              if (state?.activeLabel !== undefined) onSeek(Number(state.activeLabel));
            }}
          >
            <defs>
              <linearGradient id="occ-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={EMERALD} stopOpacity={0.45} />
                <stop offset="75%" stopColor={EMERALD} stopOpacity={0.06} />
                <stop offset="100%" stopColor={EMERALD} stopOpacity={0} />
              </linearGradient>
            </defs>
            {analytics.presenceIntervals.map((iv) => (
              <ReferenceArea
                key={`${iv[0]}-${iv[1]}`}
                x1={iv[0]}
                x2={iv[1]}
                fill={EMERALD}
                fillOpacity={0.05}
              />
            ))}
            <CartesianGrid strokeDasharray="2 6" stroke={GRID} vertical={false} />
            <XAxis
              dataKey="ts"
              type="number"
              domain={[0, domainMax]}
              tickFormatter={(v) => msToClock(Number(v))}
              stroke={AXIS}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              minTickGap={40}
            />
            <YAxis
              allowDecimals={false}
              stroke={AXIS}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              width={28}
            />
            <Tooltip
              cursor={{ stroke: EMERALD, strokeOpacity: 0.3, strokeWidth: 1 }}
              content={<ChartTooltip />}
            />
            <Area
              type="monotone"
              dataKey="students"
              stroke={EMERALD}
              strokeWidth={2.5}
              fill="url(#occ-fill)"
              activeDot={{ r: 4, fill: EMERALD, stroke: "#ffffff", strokeWidth: 2 }}
              animationDuration={700}
            />
            {peak && (
              <ReferenceDot
                x={peak.ts_ms}
                y={peak.students}
                r={5}
                fill={AMBER}
                stroke="#ffffff"
                strokeWidth={2}
                label={{
                  value: `Peak ${peak.students} @ ${msToClock(peak.ts_ms)}`,
                  position: "top",
                  fontSize: 11,
                  fill: AXIS,
                }}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Shaded bands mark when the teacher is present; the amber dot marks peak occupancy.
      </p>
    </Card>
  );
}
