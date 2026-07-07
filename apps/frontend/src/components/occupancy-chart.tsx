import type { RouterOutputs } from "@classroom/api-contracts";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import { msToClock } from "@/lib/format";

type Analytics = NonNullable<RouterOutputs["videos"]["get"]["analytics"]>;

const EMERALD = "#10b981";

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

  return (
    <Card className="p-4">
      <div className="mb-3 text-sm font-medium text-muted-foreground">
        Student occupancy over time
      </div>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
            onClick={(state: { activeLabel?: string | number }) => {
              if (state?.activeLabel !== undefined) onSeek(Number(state.activeLabel));
            }}
          >
            <defs>
              <linearGradient id="occ-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={EMERALD} stopOpacity={0.35} />
                <stop offset="100%" stopColor={EMERALD} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(161,161,170,0.14)" vertical={false} />
            {analytics.presenceIntervals.map((iv) => (
              <ReferenceArea
                key={`${iv[0]}-${iv[1]}`}
                x1={iv[0]}
                x2={iv[1]}
                fill={EMERALD}
                fillOpacity={0.07}
              />
            ))}
            <XAxis
              dataKey="ts"
              type="number"
              domain={[0, domainMax]}
              tickFormatter={(v) => msToClock(Number(v))}
              stroke="#71717a"
              fontSize={11}
              tickLine={false}
            />
            <YAxis
              allowDecimals={false}
              stroke="#71717a"
              fontSize={11}
              tickLine={false}
              width={28}
            />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #27272a",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelFormatter={(v) => msToClock(Number(v))}
              formatter={(v) => [String(v), "students"]}
            />
            <Area
              type="stepAfter"
              dataKey="students"
              stroke={EMERALD}
              strokeWidth={2}
              fill="url(#occ-fill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Shaded bands mark when the teacher is present.
      </p>
    </Card>
  );
}
