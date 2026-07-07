import { RPCHandler } from "@orpc/server/fetch";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { env } from "@api/lib/env";
import { appRouter } from "@api/router";
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

  app.get("/health", (c) => c.json({ ok: true }));
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
