import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { Suspense, lazy, useRef, useState } from "react";
import { EventsTable } from "@/components/events-table";
import { KpiCards } from "@/components/kpi-cards";
import { StatusBadge } from "@/components/status-badge";
import { TimelineStrip } from "@/components/timeline-strip";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { VideoPlayer } from "@/components/video-player";
import { ZoneEditor } from "@/components/zone-editor";
import { msToClock } from "@/lib/format";
import { API_URL, orpc, orpcClient } from "@/lib/orpc";

// recharts is heavy, so the occupancy chart loads on demand after the page mounts.
const OccupancyChart = lazy(() =>
  import("@/components/occupancy-chart").then((m) => ({ default: m.OccupancyChart })),
);

export const Route = createFileRoute("/videos/$id")({ component: VideoDetail });

function VideoDetail() {
  const { id } = Route.useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentMs, setCurrentMs] = useState(0);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);

  const { data, isLoading, isError } = useQuery({
    ...orpc.videos.get.queryOptions({ input: { id } }),
    refetchInterval: (query) => {
      const status = query.state.data?.video.status;
      return status && status !== "done" && status !== "failed" ? 2000 : false;
    },
  });

  const reanalyze = useMutation({
    mutationFn: () => orpcClient.analysis.reanalyze({ id }),
    onSuccess: () => queryClient.invalidateQueries(),
  });
  const remove = useMutation({
    mutationFn: () => orpcClient.videos.delete({ id }),
    onSuccess: async () => {
      await queryClient.invalidateQueries();
      navigate({ to: "/" });
    },
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="aspect-video w-full rounded-xl" />
        <Skeleton className="h-24 w-full rounded-xl" />
      </div>
    );
  }
  if (isError || !data) {
    return <Card className="p-6 text-sm text-destructive">Could not load this recording.</Card>;
  }

  const { video, analytics, events } = data;
  const done = video.status === "done";
  const seek = (ms: number) => {
    if (videoRef.current) videoRef.current.currentTime = ms / 1000;
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link to="/" className="text-xs text-muted-foreground hover:text-foreground">
            Back to library
          </Link>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">{video.title}</h1>
          <div className="mt-1 flex items-center gap-2 text-sm text-muted-foreground">
            <StatusBadge status={video.status} />
            <span className="tabular-nums">{msToClock(video.durationMs)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" disabled={!done} onClick={() => setEditorOpen(true)}>
            Edit zones
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={reanalyze.isPending || !done}
            onClick={() => reanalyze.mutate()}
          >
            {reanalyze.isPending ? "Re-analyzing" : "Re-analyze"}
          </Button>
          {confirmDelete ? (
            <Button
              variant="destructive"
              size="sm"
              disabled={remove.isPending}
              onClick={() => remove.mutate()}
            >
              {remove.isPending ? "Deleting" : "Confirm delete"}
            </Button>
          ) : (
            <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(true)}>
              Delete
            </Button>
          )}
        </div>
      </div>

      <VideoPlayer
        videoRef={videoRef}
        videoId={id}
        streamUrl={`${API_URL}/videos/${id}/stream`}
        width={video.width}
        height={video.height}
        analyticsReady={done}
        zones={data.zones.filter((z): z is typeof z & { polygon: [number, number][] } =>
          Array.isArray(z.polygon),
        )}
        onTimeUpdate={setCurrentMs}
      />

      {!done ? (
        <Card className="p-6 text-sm text-muted-foreground">
          Analysis {video.status}. {Math.round((video.progress ?? 0) * 100)}% complete.
        </Card>
      ) : analytics ? (
        <div className="space-y-6">
          <KpiCards analytics={analytics} durationMs={video.durationMs} />
          <TimelineStrip
            durationMs={video.durationMs}
            presenceIntervals={analytics.presenceIntervals}
            boardIntervals={analytics.boardIntervals}
            events={events}
            currentMs={currentMs}
            onSeek={seek}
          />
          <Suspense fallback={<Skeleton className="h-64 w-full rounded-xl" />}>
            <OccupancyChart analytics={analytics} durationMs={video.durationMs} onSeek={seek} />
          </Suspense>
          <div>
            <h2 className="mb-3 text-sm font-medium text-muted-foreground">Teacher events</h2>
            <EventsTable events={events} onSeek={seek} />
          </div>
        </div>
      ) : (
        <Card className="p-6 text-sm text-muted-foreground">No analytics available.</Card>
      )}

      {editorOpen && (
        <ZoneEditor
          videoId={id}
          frameSrc={video.thumbnailUrl ? `${API_URL}${video.thumbnailUrl}` : null}
          aspect={video.width && video.height ? video.width / video.height : 16 / 9}
          initialZones={data.zones}
          onClose={() => setEditorOpen(false)}
        />
      )}
    </div>
  );
}
