import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Lock } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { API_URL } from "@/lib/orpc";

/**
 * Single-tenant lock screen. The API's /auth/me says whether a password is
 * configured and whether this browser holds a valid session cookie; until
 * both are satisfied, nothing else renders (and every API call would 401
 * anyway — this is UX, the API is the enforcement).
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const me = useQuery({
    queryKey: ["auth", "me"],
    queryFn: async () => {
      const res = await fetch(`${API_URL}/auth/me`, { credentials: "include" });
      if (!res.ok) throw new Error("auth check failed");
      return (await res.json()) as { authRequired: boolean; authenticated: boolean };
    },
    staleTime: 60_000,
    retry: 1,
  });

  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fail OPEN only for rendering: if /auth/me itself is unreachable the app
  // shell still shows its own "service unreachable" states, which beats a
  // dead lock screen with no diagnostics.
  if (me.isLoading) return null;
  const data = me.data;
  if (!data || !data.authRequired || data.authenticated) return <>{children}</>;

  const submit = async () => {
    if (!password || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        credentials: "include",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!res.ok) {
        setError(res.status === 401 ? "Wrong password." : "Sign-in failed. Try again.");
        setSubmitting(false);
        return;
      }
      await queryClient.invalidateQueries();
    } catch {
      setError("Could not reach the service.");
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-dvh items-center justify-center bg-background p-6">
      <div className="reveal w-full max-w-sm space-y-6">
        <div className="space-y-2 text-center">
          <span className="font-display text-2xl font-medium tracking-[-0.02em]">Luminary</span>
          <p className="text-sm text-muted-foreground">
            Enter the admin password to open the dashboard.
          </p>
        </div>
        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <div className="relative">
            <Lock className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Admin password"
              autoFocus
              className="w-full rounded-lg border border-input bg-card py-2.5 pl-9 pr-3 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <Button type="submit" className="w-full" disabled={!password || submitting}>
            {submitting ? "Signing in…" : "Sign in"}
          </Button>
        </form>
      </div>
    </div>
  );
}
