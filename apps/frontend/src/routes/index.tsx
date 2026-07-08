import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { Clapperboard } from "lucide-react";
import type { CSSProperties } from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { UploadZone } from "@/components/upload-zone";
import { formatDate, msToClock, percentOf } from "@/lib/format";
import { API_URL, orpc } from "@/lib/orpc";

export const Route = createFileRoute("/")({ component: Library });

function Library() {
  const { data, isLoading, isError } = useQuery(orpc.videos.list.queryOptions());

  return (
    <div className="space-y-8">
      <header className="reveal space-y-1.5">
        <h1 className="font-display text-3xl font-semibold tracking-tight">Lessons</h1>
        <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">
          Upload a classroom recording and Luminary brings the teaching to light — presence, board
          time, circulation, and how the class settled.
        </p>
      </header>

      <UploadZone />

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
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
              Drop a classroom recording above to see your first analysis.
            </p>
          </div>
        </Card>
      ) : (
        <div className="stagger grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
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
                      className="size-full object-cover"
                    />
                    <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/25 to-transparent opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
                  </div>
                ) : (
                  <div className="flex aspect-video items-center justify-center bg-muted text-xs text-muted-foreground">
                    {v.status === "failed" ? "failed" : "processing…"}
                  </div>
                )}
                <div className="space-y-3 p-4">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium">{v.title}</span>
                    <Badge variant={v.status === "done" ? "default" : "outline"}>{v.status}</Badge>
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
