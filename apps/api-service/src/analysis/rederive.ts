import { getZones, replaceDerived, type ZoneInput } from "@api/db/queries";
import { mlGetJob, mlGetJobResult, mlStartRederive } from "@api/lib/ml";
import type { AnalysisResult } from "@api/lib/ml";

const POLL_MS = 5_000;
const MAX_POLLS = (2 * 60 * 60) / 5; // 2h of deriving is a hang, not a lesson

async function pollRederive(jobId: string): Promise<AnalysisResult> {
  for (let i = 0; i < MAX_POLLS; i++) {
    const status = await mlGetJob(jobId);
    if (status.status === "done") return mlGetJobResult(jobId);
    if (status.status === "failed") {
      throw new Error(`ML rederive failed: ${status.error ?? "unknown error"}`);
    }
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
  throw new Error("ML rederive did not complete within 2 hours");
}

export async function applyRederive(
  videoId: string,
  zones: ZoneInput[],
  opts: { markDone?: boolean },
): Promise<AnalysisResult> {
  // Queue it on the ML side and poll, rather than holding one long request:
  // a 37-minute derive outlives any HTTP connection (Bun drops a quiet socket
  // at 300s), and the retries silently redo work that had already succeeded.
  const jobId = await mlStartRederive(videoId, zones);
  const result = await pollRederive(jobId);
  await replaceDerived(videoId, result, opts);
  return result;
}

export function rederiveFromRaw(
  videoId: string,
  opts: { markDone?: boolean } = {},
): Promise<AnalysisResult> {
  return getZones(videoId).then((zones) =>
    applyRederive(videoId, zones, { markDone: opts.markDone ?? true }),
  );
}
