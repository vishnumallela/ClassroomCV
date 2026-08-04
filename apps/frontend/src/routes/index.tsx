import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { Clapperboard, MapPin, Plus, School } from "lucide-react";
import type { CSSProperties } from "react";
import { useState } from "react";
import { NewClassroomDialog } from "@/components/new-classroom-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDate } from "@/lib/format";
import { orpc } from "@/lib/orpc";

export const Route = createFileRoute("/")({ component: ClassroomsHome });

function ClassroomsHome() {
  const { data, isLoading, isError } = useQuery(orpc.classrooms.list.queryOptions());
  const [creating, setCreating] = useState(false);

  return (
    <div className="space-y-8">
      <header className="reveal flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-1.5">
          <span className="micro-label">Fleet overview</span>
          <h1 className="font-display text-3xl font-semibold tracking-tight">Classrooms</h1>
          <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">
            Each classroom holds its camera's zone configuration and every lesson recorded in it.
            Register a room once, then upload lessons to see presence, board time, and how the
            class settles.
          </p>
        </div>
        <Button onClick={() => setCreating(true)}>
          <Plus className="size-4" />
          New classroom
        </Button>
      </header>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-44 rounded-xl" />
          ))}
        </div>
      ) : isError ? (
        <Card className="p-6 text-sm text-destructive">Could not reach the analytics service.</Card>
      ) : !data || data.length === 0 ? (
        <Card className="flex flex-col items-center gap-4 p-14 text-center">
          <span className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <School className="size-6" />
          </span>
          <div className="space-y-1">
            <p className="font-display text-lg font-medium">Register your first classroom</p>
            <p className="mx-auto max-w-sm text-sm text-muted-foreground">
              A classroom is one room and one camera. You'll configure its board and door zones,
              then upload lessons into it.
            </p>
          </div>
          <Button onClick={() => setCreating(true)}>
            <Plus className="size-4" />
            New classroom
          </Button>
        </Card>
      ) : (
        <div className="stagger grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.map((c, i) => (
            <Link
              key={c.id}
              to="/classrooms/$id/videos"
              params={{ id: c.id }}
              style={{ "--i": i } as CSSProperties}
            >
              <Card className="group flex h-full flex-col gap-4 p-5 transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/50 hover:shadow-[0_8px_24px_-12px_oklch(0.5_0.088_167_/_0.35)]">
                <div className="flex items-start justify-between gap-3">
                  <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <School className="size-5" />
                  </span>
                  {c.processingCount > 0 ? (
                    <Badge variant="secondary" className="animate-pulse">
                      {c.processingCount} processing
                    </Badge>
                  ) : c.zoneKinds.length > 0 ? (
                    <Badge variant="outline">zones set</Badge>
                  ) : null}
                </div>
                <div className="min-w-0 space-y-1">
                  <p className="truncate font-display text-lg font-medium leading-tight">
                    {c.name}
                  </p>
                  {c.location && (
                    <p className="flex items-center gap-1 truncate text-xs text-muted-foreground">
                      <MapPin className="size-3 shrink-0" />
                      {c.location}
                    </p>
                  )}
                </div>
                <div className="mt-auto flex items-center gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1.5 tabular-nums">
                    <Clapperboard className="size-3.5" />
                    {c.videoCount} lesson{c.videoCount === 1 ? "" : "s"}
                  </span>
                  {c.lastUploadAt && <span>Last upload {formatDate(c.lastUploadAt)}</span>}
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}

      {creating && <NewClassroomDialog onClose={() => setCreating(false)} />}
    </div>
  );
}
