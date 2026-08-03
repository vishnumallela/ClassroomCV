import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { BarChart3 } from "lucide-react";
import type { CSSProperties } from "react";
import { StatusBadge } from "@/components/status-badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatDate, msToClock, percentOf } from "@/lib/format";
import { orpc } from "@/lib/orpc";

export const Route = createFileRoute("/classrooms/$id/metrics")({ component: ClassroomMetrics });

/**
 * The classroom across all its lessons: aggregate tiles on top, one row per
 * lesson underneath. Per-lesson depth (timeline, board sessions, heatmap)
 * stays on the lesson page — this view answers "how is this room doing
 * overall?".
 */
function ClassroomMetrics() {
  const { id } = Route.useParams();
  const metrics = useQuery(orpc.classrooms.metrics.queryOptions({ input: { id } }));
  const videos = useQuery(orpc.videos.list.queryOptions({ input: { classroomId: id } }));

  if (metrics.isLoading || videos.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-28 rounded-xl" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
    );
  }
  if (metrics.isError || !metrics.data) {
    return (
      <Card className="p-6 text-sm text-destructive">Could not load classroom metrics.</Card>
    );
  }

  const m = metrics.data;
  const lessons = (videos.data ?? []).filter((v) => v.status === "done");

  if (m.analyzedCount === 0) {
    return (
      <Card className="flex flex-col items-center gap-3 p-12 text-center">
        <span className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <BarChart3 className="size-6" />
        </span>
        <div className="space-y-1">
          <p className="font-display text-lg font-medium">No analyzed lessons yet</p>
          <p className="mx-auto max-w-sm text-sm text-muted-foreground">
            Metrics aggregate across every analyzed lesson in this classroom. Upload a recording
            and they'll appear here.
          </p>
        </div>
      </Card>
    );
  }

  const tiles = [
    {
      label: "Lessons analyzed",
      value: String(m.analyzedCount),
      sub: `of ${m.videoCount} uploaded`,
    },
    {
      label: "Hours analyzed",
      value: (m.totalDurationMs / 3_600_000).toFixed(1),
      sub: "of footage",
    },
    {
      label: "Time at board",
      value: m.boardTrackedMs > 0 ? percentOf(m.teacherBoardMs, m.boardTrackedMs) : "n/a",
      sub: m.boardTrackedMs > 0 ? "of board-tracked time" : "no board zone yet",
    },
    {
      label: "Teacher entries",
      value: String(m.totalEntries),
      sub: "into the room, total",
    },
    {
      label: "Teacher exits",
      value: String(m.totalExits),
      sub: "out of the room, total",
    },
    {
      label: "Teacher presence",
      value: percentOf(m.teacherPresentMs, m.totalDurationMs),
      sub: "of analyzed time",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="stagger grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        {tiles.map((t, i) => (
          <Card
            key={t.label}
            className="p-4 transition-colors hover:border-primary/40"
            style={{ "--i": i } as CSSProperties}
          >
            <div className="text-xs text-muted-foreground">{t.label}</div>
            <div className="mt-1.5 font-display text-2xl font-semibold tabular-nums tracking-tight">
              {t.value}
            </div>
            <div className="mt-0.5 text-xs text-muted-foreground">{t.sub}</div>
          </Card>
        ))}
      </div>

      <Card className="overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Lesson</TableHead>
              <TableHead>Date</TableHead>
              <TableHead className="text-right">Duration</TableHead>
              <TableHead>Teacher presence</TableHead>
              <TableHead className="text-right">Door in · out</TableHead>
              <TableHead className="text-right">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {(videos.data ?? []).map((v) => {
              const presencePct =
                v.status === "done" && v.teacherPresentMs !== null && v.durationMs
                  ? Math.min(100, Math.round((v.teacherPresentMs / v.durationMs) * 100))
                  : null;
              return (
                <TableRow key={v.id}>
                  <TableCell className="max-w-56">
                    <Link
                      to="/videos/$id"
                      params={{ id: v.id }}
                      className="block truncate font-medium underline-offset-2 hover:underline"
                    >
                      {v.title}
                    </Link>
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-muted-foreground">
                    {formatDate(v.uploadedAt)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {msToClock(v.durationMs)}
                  </TableCell>
                  <TableCell>
                    {presencePct !== null ? (
                      <span className="flex items-center gap-2">
                        <span className="h-1.5 w-24 overflow-hidden rounded-full bg-muted">
                          <span
                            className="block h-full rounded-full bg-primary"
                            style={{ width: `${presencePct}%` }}
                          />
                        </span>
                        <span className="text-xs tabular-nums text-muted-foreground">
                          {presencePct}%
                        </span>
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {v.status === "done" ? `${v.entries ?? 0} · ${v.exits ?? 0}` : "—"}
                  </TableCell>
                  <TableCell className="text-right">
                    <StatusBadge status={v.status} />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>

      {lessons.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Aggregates cover the {m.analyzedCount} analyzed lesson
          {m.analyzedCount === 1 ? "" : "s"}; board share divides by board-tracked time only, so
          lessons without a board zone never dilute it.
        </p>
      )}
    </div>
  );
}
