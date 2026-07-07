import type { QueryClient } from "@tanstack/react-query";
import { Link, Outlet, createRootRouteWithContext } from "@tanstack/react-router";

export interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
});

const navLink =
  "rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground";

function RootLayout() {
  return (
    <div className="min-h-dvh bg-background text-foreground">
      <header className="sticky top-0 z-30 border-b border-border/60 bg-background/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
          <Link to="/" className="font-semibold tracking-tight">
            classroomCV
          </Link>
          <nav className="flex items-center gap-1">
            <Link
              to="/"
              className={navLink}
              activeOptions={{ exact: true }}
              activeProps={{ className: "text-foreground" }}
            >
              Library
            </Link>
            <Link
              to="/architecture"
              className={navLink}
              activeProps={{ className: "text-foreground" }}
            >
              Architecture
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
