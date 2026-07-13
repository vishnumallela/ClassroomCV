/**
 * Durable object storage for video + thumbnail bytes, with two backends.
 *
 * - "local": bytes live under DATA_DIR (the dev default; unchanged behaviour).
 * - "s3": bytes live in MinIO / S3 / Cloudflare R2 (any S3-compatible store,
 *   selected by endpoint). On-prem MinIO keeps student video on the school's
 *   own infrastructure (a FERPA/GDPR data-residency win) while being one
 *   config flip away from cloud S3.
 *
 * The ML service and ffmpeg require a real local file path (there is an
 * SSRF/arbitrary-read guard, detector._validate_video_path, that only accepts
 * a local file inside DATA_DIR). So even with the s3 backend the canonical
 * `filePath` stays a DATA_DIR path: the worker calls `ensureLocal` to
 * materialize the object to that path before analysis, and streaming serves
 * from that local cache. S3 is the durable source of truth; DATA_DIR is a cache.
 *
 * Uses Bun's built-in S3 client (Bun.S3Client) — no extra dependency.
 */
import { S3Client } from "bun";
import { randomUUID } from "node:crypto";
import { mkdir, rename, unlink } from "node:fs/promises";
import { dirname } from "node:path";
import { env } from "@api/lib/env";

export const isS3 = env.API_SERVICE__STORAGE_BACKEND === "s3";

const DATA_DIR = env.API_SERVICE__DATA_DIR.replace(/\/+$/u, "");

let cachedClient: S3Client | undefined;
function client(): S3Client {
  if (!cachedClient) {
    cachedClient = new S3Client({
      endpoint: env.API_SERVICE__S3_ENDPOINT,
      bucket: env.API_SERVICE__S3_BUCKET,
      accessKeyId: env.API_SERVICE__S3_ACCESS_KEY,
      secretAccessKey: env.API_SERVICE__S3_SECRET_KEY,
      region: env.API_SERVICE__S3_REGION,
    });
  }
  return cachedClient;
}

/** Object key for a canonical local path (its path relative to DATA_DIR). */
export function objectKey(localPath: string): string {
  return localPath.startsWith(`${DATA_DIR}/`) ? localPath.slice(DATA_DIR.length + 1) : localPath;
}

export interface WriteSink {
  write(chunk: Uint8Array): void;
  end(): Promise<void>;
}

/**
 * Open a streaming sink for an upload. Chunks are written incrementally so a
 * multi-GB video never buffers in memory. Local: a Bun FileSink at localPath.
 * S3: a multipart writer at the derived key (nothing lands on local disk).
 */
export async function openWriteSink(localPath: string): Promise<WriteSink> {
  if (!isS3) {
    await mkdir(dirname(localPath), { recursive: true });
    const sink = Bun.file(localPath).writer();
    return {
      write: (c) => void sink.write(c),
      end: async () => void (await sink.end()),
    };
  }
  const w = client()
    .file(objectKey(localPath))
    .writer({ retry: 3, queueSize: 8, partSize: 8 * 1024 * 1024 });
  return {
    write: (c) => void w.write(c),
    end: async () => void (await w.end()),
  };
}

/** In-process single-flight: concurrent ensureLocal calls for the same path
 *  share one download instead of racing to write the same file. */
const inflightDownloads = new Map<string, Promise<void>>();

/**
 * Guarantee the bytes exist at `localPath` (probe / ffmpeg / the ML service
 * need a real file). Local: already present. S3: stream the object to disk if
 * the local cache is missing.
 *
 * The object is streamed to a private temp file and then atomically renamed
 * onto the canonical path, so (a) two concurrent cold-cache reads can never
 * interleave bytes into the served file, and (b) a transient S3 error can never
 * leave a truncated cache that is then served forever — only a fully streamed
 * object is ever published, and a failed stream unlinks its temp.
 */
export function ensureLocal(localPath: string): Promise<void> {
  if (!isS3) return Promise.resolve();
  const existing = inflightDownloads.get(localPath);
  if (existing) return existing;

  const download = (async () => {
    if (await Bun.file(localPath).exists()) return;
    await mkdir(dirname(localPath), { recursive: true });
    const tmp = `${localPath}.tmp-${randomUUID()}`;
    try {
      // Bun.write streams the S3File to disk without buffering the whole object.
      await Bun.write(tmp, client().file(objectKey(localPath)));
      await rename(tmp, localPath); // atomic within DATA_DIR (same filesystem)
    } catch (err) {
      await unlink(tmp).catch(() => undefined);
      throw err;
    }
  })();

  inflightDownloads.set(localPath, download);
  return download.finally(() => inflightDownloads.delete(localPath));
}

/**
 * Persist an already-written local file into the durable store (used for the
 * thumbnail ffmpeg writes locally). Local: no-op. S3: upload it.
 */
export async function putLocalFile(localPath: string): Promise<void> {
  if (!isS3) return;
  await client().file(objectKey(localPath)).write(Bun.file(localPath));
}

/**
 * A time-limited presigned GET URL for the object, or null on the local
 * backend. Lets ffprobe/ffmpeg read only the bytes they need (header, one
 * seeked frame) over HTTP range, and lets a remote ML worker fetch the video
 * itself, instead of downloading the whole file to the API node.
 */
export function presignGet(localPath: string, expiresInSeconds = 3600): string | null {
  if (!isS3) return null;
  return client()
    .file(objectKey(localPath))
    .presign({ method: "GET", expiresIn: expiresInSeconds });
}

/** Does the durable object exist? (local: the file; s3: the object) */
export function exists(localPath: string): Promise<boolean> {
  if (!isS3) return Bun.file(localPath).exists();
  return client().exists(objectKey(localPath));
}

/** Delete the durable objects for the given canonical local paths (s3 only;
 *  local cleanup is a filesystem rm handled by the caller). */
export async function removeObjects(localPaths: string[]): Promise<void> {
  if (!isS3) return;
  await Promise.all(
    localPaths.filter(Boolean).map((p) =>
      client()
        .file(objectKey(p))
        .delete()
        .catch(() => undefined),
    ),
  );
}
