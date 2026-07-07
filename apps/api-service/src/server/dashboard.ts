import { createBullBoard } from "@bull-board/api";
import { BullMQAdapter } from "@bull-board/api/bullMQAdapter";
import { HonoAdapter } from "@bull-board/hono";
import { Hono } from "hono";
import { serveStatic } from "hono/bun";
import { queues } from "@api/lib/queue";

export function createDashboard(basePath: string): Hono {
  const serverAdapter = new HonoAdapter(serveStatic);
  createBullBoard({
    queues: Object.values(queues).map((queue) => new BullMQAdapter(queue)),
    serverAdapter,
  });
  serverAdapter.setBasePath(basePath);

  const app = new Hono();
  app.route("/", serverAdapter.registerPlugin());
  return app;
}
