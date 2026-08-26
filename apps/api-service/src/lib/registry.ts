import { logger } from "@api/lib/logger";

/**
 * Read an image's manifest and config straight from its registry, so the app
 * can say WHAT it is about to run before it rents a GPU to run it.
 *
 * This exists because a tag is not a description. `:latest` on this project's
 * registry is written only by `main`, so while the RF-DETR rewrite sat on a
 * branch, `:latest` was still the previous YOLO/identity-stack service — same
 * name, same port, same `/health`, entirely different program. Pointing a pod
 * at it produces a container that starts, reports RUNNING, ignores every
 * `RFDETR_*` variable it is handed (the settings model is `extra="ignore"`),
 * and has no entrypoint and therefore no sshd to diagnose it with. The GPU
 * bills the whole time.
 *
 * The config blob settles it in one request: the RF-DETR image declares
 * `RFDETR_WEIGHTS`, the old one declares `MODEL_NAME`. That is a positive test
 * for the right program rather than a guess from the tag name.
 */

/**
 * The outcome of an inspection, kept distinct because the three cases demand
 * different responses. `not-found` is a certainty — that tag cannot be pulled,
 * so a pod on it burns boot time and fails. `unknown` is an absence of
 * evidence: a private package, a registry that does not speak the token flow,
 * a timeout. Collapsing the two would either block valid private images or
 * wave through a typo, and the typo costs a rented GPU that never starts.
 */
export type ImageCheck =
  | { status: "ok"; info: ImageInfo; message: null }
  | { status: "not-found"; info: null; message: string }
  | { status: "unknown"; info: null; message: string };

export interface ImageInfo {
  /** Registry host, e.g. "ghcr.io". */
  registry: string;
  repository: string;
  reference: string;
  /** Immutable content digest the tag currently resolves to. */
  digest: string | null;
  /** Build time from the image config (UTC ISO), when the builder set one. */
  createdAt: string | null;
  entrypoint: string[] | null;
  cmd: string[] | null;
  /** Image-declared env, minus the base image's CUDA/Python noise. */
  env: Record<string, string>;
  /** Whether the image declares the RF-DETR service's own variables. */
  isMlService: boolean;
}

/** Base-image variables that say nothing about which program this is. */
const NOISE = /^(PATH|LD_LIBRARY_PATH|LANG|LC_|PYTHON|PYTORCH_VERSION|NV_|NVIDIA_|CUDA_)/u;

/**
 * Split "ghcr.io/owner/repo/name:tag" into its parts.
 *
 * A leading segment counts as a registry only when it looks like a host (has a
 * dot, a colon, or is "localhost") — otherwise "owner/name" would be read as
 * registry "owner", which is how Docker Hub short names are meant to resolve.
 */
export function parseImageRef(image: string): {
  registry: string;
  repository: string;
  reference: string;
} {
  let rest = image.trim();
  let registry = "docker.io";

  const slash = rest.indexOf("/");
  if (slash > 0) {
    const head = rest.slice(0, slash);
    if (head.includes(".") || head.includes(":") || head === "localhost") {
      registry = head;
      rest = rest.slice(slash + 1);
    }
  }

  // A digest wins over a tag; a colon after the last slash is the tag.
  let reference = "latest";
  const at = rest.indexOf("@");
  if (at > 0) {
    reference = rest.slice(at + 1);
    rest = rest.slice(0, at);
  } else {
    const colon = rest.lastIndexOf(":");
    if (colon > rest.lastIndexOf("/")) {
      reference = rest.slice(colon + 1);
      rest = rest.slice(0, colon);
    }
  }

  // Docker Hub official images live under library/.
  if (registry === "docker.io" && !rest.includes("/")) rest = `library/${rest}`;
  return { registry, repository: rest, reference };
}

/**
 * The host that actually serves the registry API for a given image host.
 * "docker.io" is the name in an image reference but has never served /v2 —
 * that is registry-1.docker.io, and asking docker.io returns a web page, which
 * reads as "cannot inspect" rather than the real answer.
 */
function apiHost(registry: string): string {
  return registry === "docker.io" || registry === "index.docker.io"
    ? "registry-1.docker.io"
    : registry;
}

const MANIFEST_ACCEPT = [
  "application/vnd.oci.image.index.v1+json",
  "application/vnd.docker.distribution.manifest.list.v2+json",
  "application/vnd.oci.image.manifest.v1+json",
  "application/vnd.docker.distribution.manifest.v2+json",
].join(",");

/**
 * Fetch with the registry's own auth dance: try anonymously, and on a 401 read
 * the `WWW-Authenticate` challenge, get a token from the realm it names, and
 * retry. This is the standard flow, so it works for GHCR, Docker Hub and most
 * others without special-casing any of them.
 */
async function registryFetch(
  url: string,
  accept: string,
  token: { value: string | null },
  signal: AbortSignal,
): Promise<Response> {
  const headers = (): HeadersInit =>
    token.value ? { accept, authorization: `Bearer ${token.value}` } : { accept };

  let res = await fetch(url, { headers: headers(), signal });
  if (res.status !== 401) return res;

  const challenge = res.headers.get("www-authenticate") ?? "";
  const field = (name: string) => new RegExp(`${name}="([^"]+)"`, "u").exec(challenge)?.[1];
  const realm = field("realm");
  if (!realm) return res;

  const tokenUrl = new URL(realm);
  const service = field("service");
  const scope = field("scope");
  if (service) tokenUrl.searchParams.set("service", service);
  if (scope) tokenUrl.searchParams.set("scope", scope);

  const auth = await fetch(tokenUrl, { signal });
  if (!auth.ok) return res;
  const body = (await auth.json()) as { token?: string; access_token?: string };
  token.value = body.token ?? body.access_token ?? null;
  if (!token.value) return res;

  res = await fetch(url, { headers: headers(), signal });
  return res;
}

const unknown = (message: string): ImageCheck => ({ status: "unknown", info: null, message });

/**
 * Inspect an image. See ImageCheck for why the failure cases are distinguished
 * rather than merged into a single null.
 */
export async function inspectImage(image: string, timeoutMs = 8000): Promise<ImageCheck> {
  const { registry, repository, reference } = parseImageRef(image);
  const base = `https://${apiHost(registry)}/v2/${repository}`;
  const token = { value: null as string | null };
  const signal = AbortSignal.timeout(timeoutMs);

  try {
    const res = await registryFetch(
      `${base}/manifests/${reference}`,
      MANIFEST_ACCEPT,
      token,
      signal,
    );
    if (!res.ok) {
      logger.info({ image, status: res.status }, "image inspect: registry declined");
      // 404 is the registry answering clearly: no such repository or tag.
      // 401/403 after the token dance means private, which is not our business
      // to judge — RunPod may hold credentials this app does not.
      return res.status === 404
        ? {
            status: "not-found",
            info: null,
            message: `${registry} has no "${reference}" for ${repository}.`,
          }
        : unknown(`${registry} would not describe this image (HTTP ${res.status}).`);
    }
    const digest = res.headers.get("docker-content-digest");
    let manifest = (await res.json()) as {
      config?: { digest: string };
      manifests?: { digest: string; platform?: { os?: string; architecture?: string } }[];
    };

    // Multi-arch index: descend into the linux/amd64 image. RunPod pods are
    // amd64, so that is the one whose config actually describes what will run.
    if (manifest.manifests?.length) {
      const child =
        manifest.manifests.find(
          (m) => m.platform?.os === "linux" && m.platform?.architecture === "amd64",
        ) ?? manifest.manifests[0];
      if (!child) return unknown("The image index listed no usable platform.");
      const sub = await registryFetch(
        `${base}/manifests/${child.digest}`,
        MANIFEST_ACCEPT,
        token,
        signal,
      );
      if (!sub.ok) return unknown(`Could not read the linux/amd64 manifest (HTTP ${sub.status}).`);
      manifest = (await sub.json()) as typeof manifest;
    }

    const configDigest = manifest.config?.digest;
    if (!configDigest) return unknown("The manifest carried no image config.");

    const blob = await registryFetch(`${base}/blobs/${configDigest}`, "*/*", token, signal);
    if (!blob.ok) return unknown(`Could not read the image config (HTTP ${blob.status}).`);
    const config = (await blob.json()) as {
      created?: string;
      config?: { Env?: string[]; Entrypoint?: string[]; Cmd?: string[] };
    };

    const env: Record<string, string> = {};
    for (const entry of config.config?.Env ?? []) {
      const eq = entry.indexOf("=");
      if (eq <= 0) continue;
      const key = entry.slice(0, eq);
      if (!NOISE.test(key)) env[key] = entry.slice(eq + 1);
    }

    return {
      status: "ok",
      message: null,
      info: {
        registry,
        repository,
        reference,
        digest,
        createdAt: config.created ?? null,
        entrypoint: config.config?.Entrypoint ?? null,
        cmd: config.config?.Cmd ?? null,
        env,
        isMlService: "RFDETR_WEIGHTS" in env,
      },
    };
  } catch (err) {
    logger.info({ image, err }, "image inspect failed");
    return unknown("The registry could not be reached.");
  }
}
