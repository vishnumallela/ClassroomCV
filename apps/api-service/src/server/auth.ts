import { timingSafeEqual } from "node:crypto";
import type { Context, Hono, Next } from "hono";
import { deleteCookie, getSignedCookie, setSignedCookie } from "hono/cookie";
import { env } from "@api/lib/env";

/**
 * Single-tenant admin auth. API_SERVICE__ADMIN_PASSWORD gates every route
 * except /health and /auth/*; empty password (dev default) disables the gate.
 *
 * Sessions are a signed cookie whose HMAC secret IS the admin password, so
 * rotating the password invalidates every session with zero stored state.
 *
 * httpOnly + SameSite=Lax + Secure. Lax, not None, because the SPA reaches
 * the API through its own origin (Caddy proxies /api — see
 * apps/frontend/Caddyfile): the cookie is first-party, which is the only way
 * it survives a browser at all, and Lax then blocks cross-site requests from
 * riding on it. A deployment that serves the API on its own domain instead
 * gets a 200 from /auth/login and a session that never sticks.
 */

const COOKIE = "luminary_session";
const COOKIE_VALUE = "ok";
const MAX_AGE_S = 30 * 24 * 60 * 60; // re-login monthly

function passwordConfigured(): boolean {
  return env.API_SERVICE__ADMIN_PASSWORD.length > 0;
}

function safeEqual(a: string, b: string): boolean {
  const ab = Buffer.from(a);
  const bb = Buffer.from(b);
  // Compare same-length buffers only; length leak is fine (it's a password
  // form, not an oracle).
  return ab.length === bb.length && timingSafeEqual(ab, bb);
}

async function hasValidSession(c: Context): Promise<boolean> {
  const value = await getSignedCookie(c, env.API_SERVICE__ADMIN_PASSWORD, COOKIE);
  return value === COOKIE_VALUE;
}

/** Gate middleware: 401 JSON for anything unauthenticated. */
export async function requireAuth(c: Context, next: Next): Promise<Response | void> {
  if (!passwordConfigured()) return next();
  if (c.req.method === "OPTIONS") return next(); // CORS preflight
  const path = new URL(c.req.url).pathname;
  if (path === "/health" || path.startsWith("/auth/")) return next();
  if (await hasValidSession(c)) return next();
  return c.json({ error: "Authentication required." }, 401);
}

export function registerAuthRoutes(app: Hono): void {
  // Who am I: lets the SPA decide between the lock screen and the app.
  app.get("/auth/me", async (c) => {
    const required = passwordConfigured();
    return c.json({
      authRequired: required,
      authenticated: required ? await hasValidSession(c) : true,
    });
  });

  app.post("/auth/login", async (c) => {
    if (!passwordConfigured()) return c.json({ ok: true });
    const body = (await c.req.json().catch(() => ({}))) as { password?: string };
    if (!body.password || !safeEqual(body.password, env.API_SERVICE__ADMIN_PASSWORD)) {
      return c.json({ error: "Wrong password." }, 401);
    }
    await setSignedCookie(c, COOKIE, COOKIE_VALUE, env.API_SERVICE__ADMIN_PASSWORD, {
      httpOnly: true,
      secure: true,
      sameSite: "Lax",
      path: "/",
      maxAge: MAX_AGE_S,
    });
    return c.json({ ok: true });
  });

  app.post("/auth/logout", (c) => {
    deleteCookie(c, COOKIE, { path: "/", secure: true, sameSite: "Lax" });
    return c.json({ ok: true });
  });
}
