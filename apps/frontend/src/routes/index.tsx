import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { UploadZone } from "@/components/upload-zone";
import { formatDate, msToClock } from "@/lib/format";
import { API_URL, orpc } from "@/lib/orpc";

export const Route = createFileRoute("/")({ component: Library });

function Library() {
  const { data, isLoading, isError } = useQuery(orpc.videos.list.queryOptions());

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Library</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Classroom recordings and their teacher and student analytics.
        </p>
      </div>

      <UploadZone />

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-40" />
          ))}
        </div>
      ) : isError ? (
        <Card className="p-6 text-sm text-destructive">Could not reach the analytics service.</Card>
      ) : !data || data.length === 0 ? (
        <Card className="p-10 text-center text-sm text-muted-foreground">No recordings yet.</Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.map((v) => (
            <Link key={v.id} to="/videos/$id" params={{ id: v.id }}>
              <Card className="overflow-hidden transition-colors hover:border-primary/50">
                {v.thumbnailUrl ? (
                  <img
                    src={`${API_URL}${v.thumbnailUrl}`}
                    alt={v.title}
                    className="aspect-video w-full bg-muted object-cover"
                  />
                ) : (
                  <div className="flex aspect-video items-center justify-center bg-muted text-xs text-muted-foreground">
                    {v.status === "failed" ? "failed" : "processing"}
                  </div>
                )}
                <div className="space-y-2 p-4">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium">{v.title}</span>
                    <Badge variant={v.status === "done" ? "default" : "outline"}>{v.status}</Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {msToClock(v.durationMs)} · {formatDate(v.uploadedAt)}
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
