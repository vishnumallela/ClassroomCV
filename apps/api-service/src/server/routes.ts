import { rm } from "node:fs/promises";
import { basename, extname, join } from "node:path";
import type { Context, Hono } from "hono";
import { createVideo, deleteVideoRows, getVideo, setWorkflowRunId } from "@api/db/queries";
import { env } from "@api/lib/env";
import { logger } from "@api/lib/logger";
import { enqueueAnalysis } from "@api/lib/queue";
import { ensureLocal, openWriteSink, removeObjects } from "@api/lib/storage";

const CONTENT_TYPES: Record<string, string> = {
  ".mp4": "video/mp4",
  ".m4v": "video/mp4",
  ".webm": "video/webm",
  ".mov": "video/quicktime",
  ".mkv": "video/x-matroska",
  ".avi": "video/x-msvideo",
  ".ogv": "video/ogg",
};

function contentTypeFor(path: string): string {
  return CONTENT_TYPES[extname(path).toLowerCase()] ?? "video/mp4";
}

function sanitizeExtension(filename: string): string {
  const ext = extname(filename).toLowerCase();
  return /^\.[a-z0-9]{1,8}$/.test(ext) ? ext : ".mp4";
}

async function handleUpload(c: Context): Promise<Response> {
  const rawName = basename(c.req.query("filename") ?? "video.mp4") || "video.mp4";
  const ext = sanitizeExtension(rawName);
  const title = rawName.replace(/\.[^.]+$/, "").trim() || rawName;
  const id = crypto.randomUUID();
  const dir = join(env.API_SERVICE__DATA_DIR, "videos", id);
  const filePath = join(dir, `original${ext}`);

  await createVideo({ id, title, originalFilename: rawName, filePath });

  const cleanup = async (): Promise<void> => {
    await rm(dir, { recursive: true, force: true }).catch(() => undefined);
    await removeObjects([filePath]).catch(() => undefined);
    await deleteVideoRows(id).catch(() => undefined);
  };

  const body = c.req.raw.body;
  if (!body) {
    await cleanup();
    return c.json({ error: "Empty request body." }, 400);
  }

  const max = env.API_SERVICE__MAX_UPLOAD_BYTES;
  const declared = Number(c.req.header("content-length") ?? "0");
  if (declared > max) {
    await cleanup();
    return c.json({ error: "Upload exceeds size limit." }, 413);
  }

  const sink = await openWriteSink(filePath);
  let total = 0;
  let tooLarge = false;
  const reader = body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > max) {
        tooLarge = true;
        break;
      }
      sink.write(value);
    }
    await sink.end();
  } catch (err) {
    await Promise.resolve(sink.end()).catch(() => undefined);
    await cleanup();
    logger.error({ err, id }, "upload stream failed");
    return c.json({ error: "Upload failed." }, 500);
  }

  if (tooLarge) {
    await cleanup();
    return c.json({ error: "Upload exceeds size limit." }, 413);
  }

  // A DELETE may have raced the upload; if the row is gone, discard the bytes.
  if (!(await getVideo(id))) {
    await rm(dir, { recursive: true, force: true }).catch(() => undefined);
    await removeObjects([filePath]).catch(() => undefined);
    return c.json({ error: "Video was deleted during upload." }, 409);
  }

  const job = await enqueueAnalysis({ videoId: id });
  await setWorkflowRunId(id, String(job.id));
  return c.json({ id }, 201);
}

async function serveVideo(c: Context, headOnly: boolean): Promise<Response> {
  const video = await getVideo(c.req.param("id"));
  if (!video) return c.notFound();
  // s3 backend: pull the object into the local cache before range-serving it.
  await ensureLocal(video.filePath).catch(() => undefined);
  const file = Bun.file(video.filePath);
  if (!(await file.exists())) return c.notFound();

  const size = file.size;
  const contentType = contentTypeFor(video.filePath);
  const range = c.req.header("range");

  const full = (): Response =>
    new Response(headOnly ? null : file.stream(), {
      status: 200,
      headers: {
        "content-type": contentType,
        "content-length": String(size),
        "accept-ranges": "bytes",
      },
    });

  if (!range) return full();
  const match = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
  if (!match) return full();

  const unsatisfiable = (): Response =>
    new Response(null, {
      status: 416,
      headers: { "content-range": `bytes */${size}`, "accept-ranges": "bytes" },
    });

  let start: number;
  let end: number;
  if (match[1] === "") {
    const suffix = parseInt(match[2] ?? "", 10);
    if (!Number.isFinite(suffix) || suffix <= 0 || size === 0) return unsatisfiable();
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = parseInt(match[1] ?? "", 10);
    end = match[2] === "" ? size - 1 : Math.min(parseInt(match[2] ?? "", 10), size - 1);
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start >= size || start > end) {
    return unsatisfiable();
  }

  return new Response(headOnly ? null : file.slice(start, end + 1).stream(), {
    status: 206,
    headers: {
      "content-type": contentType,
      "content-length": String(end - start + 1),
      "content-range": `bytes ${start}-${end}/${size}`,
      "accept-ranges": "bytes",
    },
  });
}

async function serveThumbnail(c: Context, headOnly: boolean): Promise<Response> {
  const video = await getVideo(c.req.param("id"));
  if (!video?.thumbnailPath) return c.notFound();
  await ensureLocal(video.thumbnailPath).catch(() => undefined);
  const file = Bun.file(video.thumbnailPath);
  if (!(await file.exists())) return c.notFound();
  return new Response(headOnly ? null : file.stream(), {
    headers: { "content-type": "image/jpeg", "cache-control": "public, max-age=60" },
  });
}

export function registerBinaryRoutes(app: Hono): void {
  app.post("/videos", handleUpload);
  app.get("/videos/:id/stream", (c) => serveVideo(c, false));
  app.on("HEAD", "/videos/:id/stream", (c) => serveVideo(c, true));
  app.get("/videos/:id/thumbnail", (c) => serveThumbnail(c, false));
  app.on("HEAD", "/videos/:id/thumbnail", (c) => serveThumbnail(c, true));
}
