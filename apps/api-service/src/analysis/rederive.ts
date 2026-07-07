import { getZones, replaceDerived, type ZoneInput } from "@api/db/queries";
import { mlRederive } from "@api/lib/ml";
import type { AnalysisResult } from "@api/lib/ml";

export async function applyRederive(
  videoId: string,
  zones: ZoneInput[],
  opts: { markDone?: boolean },
): Promise<AnalysisResult> {
  const result = await mlRederive(videoId, zones);
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
