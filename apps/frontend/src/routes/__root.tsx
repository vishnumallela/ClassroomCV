import type { QueryClient } from "@tanstack/react-query";
import { Link, Outlet, createRootRouteWithContext } from "@tanstack/react-router";
import { BookOpen, LayoutGrid, ShieldCheck } from "lucide-react";
import { LuminaryLogo } from "@/components/luminary-logo";
import { ThemeToggle } from "@/components/theme-toggle";

export interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
});

const NAV = [
  { to: "/" as const, label: "Lessons", icon: LayoutGrid, exact: true },
  { to: "/architecture" as const, label: "How it works", icon: BookOpen, exact: false },
];

function RootLayout() {
  return (
    <div className="flex min-h-dvh bg-background text-foreground">
      <aside className="sticky top-0 hidden h-dvh w-64 shrink-0 flex-col border-r border-border bg-card/50 px-4 pb-4 md:flex">
        <Link to="/" className="flex h-16 items-center">
          <LuminaryLogo bloom />
        </Link>

        <nav className="mt-2 flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon, exact }) => (
            <Link
              key={to}
              to={to}
              activeOptions={{ exact }}
              className="group flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors duration-150"
              inactiveProps={{
                className: "text-muted-foreground hover:bg-accent hover:text-foreground",
              }}
              activeProps={{
                className: "bg-accent font-medium text-foreground",
              }}
            >
              <Icon className="size-[1.05rem]" />
              {label}
            </Link>
          ))}
        </nav>

        <div className="mt-auto space-y-3">
          <div className="rounded-lg border border-border bg-background/60 p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-foreground">
              <ShieldCheck className="size-4 text-primary" />
              Privacy by design
            </div>
            <p className="mt-1.5 text-[0.7rem] leading-relaxed text-muted-foreground">
              No facial recognition. No named students. Aggregate insights only.
            </p>
          </div>
          <div className="flex items-center justify-between px-1">
            <span className="font-display text-xs italic text-muted-foreground">
              Every lesson, brought to light
            </span>
            <ThemeToggle />
          </div>
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="fixed inset-x-0 top-0 z-[var(--z-sticky)] flex h-14 items-center justify-between border-b border-border bg-card/80 px-4 backdrop-blur md:hidden">
        <Link to="/">
          <LuminaryLogo />
        </Link>
        <div className="flex items-center gap-1">
          {NAV.map(({ to, label, icon: Icon, exact }) => (
            <Link
              key={to}
              to={to}
              activeOptions={{ exact }}
              aria-label={label}
              className="inline-flex size-9 items-center justify-center rounded-md text-muted-foreground"
              activeProps={{ className: "bg-accent text-foreground" }}
            >
              <Icon className="size-[1.05rem]" />
            </Link>
          ))}
          <ThemeToggle />
        </div>
      </header>

      <main className="min-w-0 flex-1 pt-14 md:pt-0">
        <div className="mx-auto max-w-6xl px-5 py-8 sm:px-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
