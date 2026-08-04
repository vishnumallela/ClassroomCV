import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { Clapperboard } from "lucide-react";
import type { CSSProperties } from "react";
import { StatusBadge } from "@/components/status-badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { UploadZone } from "@/components/upload-zone";
import { formatDate, msToClock, percentOf } from "@/lib/format";
import { API_URL, orpc } from "@/lib/orpc";

export const Route = createFileRoute("/classrooms/$id/videos")({ component: ClassroomVideos });

function ClassroomVideos() {
  const { id } = Route.useParams();
  const { data, isLoading, isError } = useQuery({
    ...orpc.videos.list.queryOptions({ input: { classroomId: id } }),
    // Poll while any lesson is still moving through the RunPod pipeline so
    // progress lands without a refresh; go quiet once everything settles.
    refetchInterval: (query) =>
      query.state.data?.some((v) => v.status !== "done" && v.status !== "failed") ? 2500 : false,
  });

  return (
    <div className="space-y-6">
      <UploadZone classroomId={id} />

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-52 rounded-xl" />
          ))}
        </div>
      ) : isError ? (
        <Card className="p-6 text-sm text-destructive">Could not reach the analytics service.</Card>
      ) : !data || data.length === 0 ? (
        <Card className="flex flex-col items-center gap-3 p-12 text-center">
          <span className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Clapperboard className="size-6" />
          </span>
          <div className="space-y-1">
            <p className="font-display text-lg font-medium">No lessons yet</p>
            <p className="text-sm text-muted-foreground">
              Drop a recording above — it uploads, processes on the GPU, and the metrics appear
              here.
            </p>
          </div>
        </Card>
      ) : (
        <div className="stagger grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {data.map((v, i) => (
            <Link
              key={v.id}
              to="/videos/$id"
              params={{ id: v.id }}
              style={{ "--i": i } as CSSProperties}
            >
              <Card className="group overflow-hidden transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/50 hover:shadow-[0_8px_24px_-12px_oklch(0.5_0.088_167_/_0.35)]">
                {v.thumbnailUrl ? (
                  <div className="relative aspect-video w-full overflow-hidden bg-muted">
                    <img
                      src={`${API_URL}${v.thumbnailUrl}`}
                      alt={v.title}
                      crossOrigin="use-credentials"
                      className="size-full object-cover"
                    />
                    <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/25 to-transparent opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
                  </div>
                ) : (
                  <div className="flex aspect-video flex-col items-center justify-center gap-2 bg-muted text-xs text-muted-foreground">
                    {v.status === "failed" ? (
                      "failed"
                    ) : (
                      <>
                        <span>processing…</span>
                        <span className="h-1 w-32 overflow-hidden rounded-full bg-border">
                          <span
                            className="block h-full rounded-full bg-primary transition-all duration-500"
                            style={{ width: `${Math.round(v.progress * 100)}%` }}
                          />
                        </span>
                      </>
                    )}
                  </div>
                )}
                <div className="space-y-3 p-4">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium">{v.title}</span>
                    <StatusBadge status={v.status} />
                  </div>
                  {v.status === "done" && v.teacherPresentMs !== null ? (
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span className="tabular-nums">
                        Teacher {percentOf(v.teacherPresentMs, v.durationMs)}
                      </span>
                      <span className="tabular-nums">
                        {v.entries ?? 0} in · {v.exits ?? 0} out
                      </span>
                    </div>
                  ) : (
                    <div className="text-xs text-muted-foreground">
                      {msToClock(v.durationMs)} · {formatDate(v.uploadedAt)}
                    </div>
                  )}
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
