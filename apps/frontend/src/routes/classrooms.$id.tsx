import { useQuery } from "@tanstack/react-query";
import { Link, Outlet, createFileRoute } from "@tanstack/react-router";
import { ArrowLeft, BarChart3, CalendarCheck, Clapperboard, MapPin, Settings2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { orpc } from "@/lib/orpc";

export const Route = createFileRoute("/classrooms/$id")({ component: ClassroomLayout });

const TABS = [
  { to: "/classrooms/$id/videos" as const, label: "Lessons", icon: Clapperboard },
  { to: "/classrooms/$id/register" as const, label: "Register", icon: CalendarCheck },
  { to: "/classrooms/$id/metrics" as const, label: "Metrics", icon: BarChart3 },
  { to: "/classrooms/$id/settings" as const, label: "Configuration", icon: Settings2 },
];

/**
 * Classroom shell: header + secondary sidebar. Everything inside a classroom
 * (lessons, aggregate metrics, zone configuration) renders in the outlet, so
 * the room context never leaves the screen.
 */
function ClassroomLayout() {
  const { id } = Route.useParams();
  const { data, isLoading, isError } = useQuery(
    orpc.classrooms.get.queryOptions({ input: { id } }),
  );

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
    );
  }
  if (isError || !data) {
    return (
      <Card className="p-6 text-sm text-destructive">
        Classroom not found.{" "}
        <Link to="/" className="underline underline-offset-2">
          Back to classrooms
        </Link>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <header className="reveal space-y-3">
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="size-3.5" />
          Classrooms
        </Link>
        <div className="space-y-1">
          <h1 className="font-display text-2xl font-semibold tracking-tight sm:text-3xl">
            {data.name}
          </h1>
          {(data.location || data.description) && (
            <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted-foreground">
              {data.location && (
                <span className="inline-flex items-center gap-1">
                  <MapPin className="size-3.5" />
                  {data.location}
                </span>
              )}
              {data.description && <span>{data.description}</span>}
            </p>
          )}
        </div>
      </header>

      <div className="flex flex-col gap-6 lg:flex-row">
        {/* Secondary sidebar: the classroom's own navigation. */}
        <nav className="flex shrink-0 gap-1 overflow-x-auto lg:sticky lg:top-8 lg:h-fit lg:w-48 lg:flex-col">
          {TABS.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              params={{ id }}
              className="flex items-center gap-2.5 whitespace-nowrap rounded-lg px-3 py-2 text-sm transition-colors duration-150"
              inactiveProps={{
                className: "text-muted-foreground hover:bg-accent hover:text-foreground",
              }}
              activeProps={{ className: "bg-accent font-medium text-foreground" }}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          ))}
        </nav>

        <div className="min-w-0 flex-1">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
