import type { QueryClient } from "@tanstack/react-query";
import { Link, Outlet, createRootRouteWithContext } from "@tanstack/react-router";
import { LayoutGrid, Network } from "lucide-react";

export interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
});

const NAV = [
  { to: "/" as const, label: "Library", icon: LayoutGrid, exact: true },
  { to: "/architecture" as const, label: "Architecture", icon: Network, exact: false },
];

function RootLayout() {
  return (
    <div className="flex min-h-dvh bg-background text-foreground">
      <aside className="sticky top-0 flex h-dvh w-60 shrink-0 flex-col gap-2 border-r border-border bg-muted/30 px-3 pb-4">
        <div className="flex h-14 items-center px-2 font-semibold tracking-tight">classroomCV</div>
        <nav className="flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon, exact }) => (
            <Link
              key={to}
              to={to}
              activeOptions={{ exact }}
              className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors"
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
      </aside>
      <main className="min-w-0 flex-1">
        <div className="mx-auto max-w-6xl px-8 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
