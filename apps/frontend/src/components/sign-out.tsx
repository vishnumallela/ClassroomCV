import { useQuery, useQueryClient } from "@tanstack/react-query";
import { LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { API_URL } from "@/lib/orpc";

/** Sidebar sign-out: rendered only when the deployment actually has auth
 * (shares AuthGate's cached /auth/me query, so it costs no extra request). */
export function SignOut() {
  const queryClient = useQueryClient();
  const me = useQuery<{ authRequired: boolean; authenticated: boolean }>({
    queryKey: ["auth", "me"],
    enabled: false, // AuthGate owns the fetch; we only read its cache
  });
  if (!me.data?.authRequired) return null;

  const signOut = async () => {
    await fetch(`${API_URL}/auth/logout`, { method: "POST", credentials: "include" }).catch(
      () => undefined,
    );
    await queryClient.invalidateQueries();
  };

  return (
    <Button variant="ghost" size="icon" aria-label="Sign out" title="Sign out" onClick={signOut}>
      <LogOut className="size-[1.05rem]" />
    </Button>
  );
}
