import { Link } from "@tanstack/react-router";
import { CircleAlert, Compass } from "lucide-react";
import { Button } from "@/components/ui/button";

/** Router-level fallbacks: a crash in any route renders this instead of a
 * white screen, and unknown URLs get a way home. */
export function RouteErrorFallback({ error }: { error: Error }) {
  return (
    <div className="flex min-h-[60vh] items-center justify-center p-6">
      <div className="w-full max-w-md space-y-4 text-center">
        <span className="mx-auto flex size-12 items-center justify-center rounded-xl bg-destructive/10 text-destructive">
          <CircleAlert className="size-6" />
        </span>
        <div className="space-y-1">
          <p className="font-display text-lg font-medium">Something went wrong</p>
          <p className="text-sm text-muted-foreground">
            {error.message || "An unexpected error occurred while rendering this page."}
          </p>
        </div>
        <div className="flex justify-center gap-2">
          <Button size="sm" onClick={() => window.location.reload()}>
            Reload
          </Button>
          <Button size="sm" variant="outline" asChild>
            <Link to="/">Back to classrooms</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

export function RouteNotFound() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center p-6">
      <div className="space-y-4 text-center">
        <span className="mx-auto flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <Compass className="size-6" />
        </span>
        <div className="space-y-1">
          <p className="font-display text-lg font-medium">Page not found</p>
          <p className="text-sm text-muted-foreground">That link doesn't lead anywhere.</p>
        </div>
        <Button size="sm" asChild>
          <Link to="/">Back to classrooms</Link>
        </Button>
      </div>
    </div>
  );
}
