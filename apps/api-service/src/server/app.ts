import { RPCHandler } from "@orpc/server/fetch";
import { Hono } from "hono";
import { basicAuth } from "hono/basic-auth";
import { cors } from "hono/cors";
import { env } from "@api/lib/env";
import { appRouter } from "@api/router";
import { registerAuthRoutes, requireAuth } from "@api/server/auth";
import { createDashboard } from "@api/server/dashboard";
import { registerBinaryRoutes } from "@api/server/routes";

const rpcHandler = new RPCHandler(appRouter);

export function createApp(): Hono {
  const app = new Hono();

  app.use(
    "*",
    cors({
      origin: env.API_SERVICE__CORS_ORIGINS,
      credentials: true,
      allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
      allowHeaders: ["Content-Type", "Authorization"],
    }),
  );
  // Everything below (RPC, uploads, media, dashboard) sits behind the admin
  // session when API_SERVICE__ADMIN_PASSWORD is set.
  app.use("*", requireAuth);
  registerAuthRoutes(app);

  app.get("/health", (c) => c.json({ ok: true }));
  // The queue dashboard can retry/discard jobs — never expose it unauthenticated.
  // Credentials come from API_SERVICE__QUEUE_DASHBOARD_USER/PASSWORD (change the
  // defaults in production).
  app.use(
    "/admin/queues/*",
    basicAuth({
      username: env.API_SERVICE__QUEUE_DASHBOARD_USER,
      password: env.API_SERVICE__QUEUE_DASHBOARD_PASSWORD,
    }),
  );
  app.route("/admin/queues", createDashboard("/admin/queues"));
  registerBinaryRoutes(app);

  app.use("/rpc/*", async (c, next) => {
    const { matched, response } = await rpcHandler.handle(c.req.raw, {
      prefix: "/rpc",
      context: {},
    });
    if (matched && response) return response;
    return next();
  });

  return app;
}
