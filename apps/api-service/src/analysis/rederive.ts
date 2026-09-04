import { getVideo, getZones, replaceDerived, type ZoneInput } from "@api/db/queries";
import { mlRederive } from "@api/lib/ml";
import type { AnalysisResult } from "@api/lib/ml";
import { schoolTimezone } from "@api/lib/school-time";
import { periodOffsetsFor } from "@api/lib/timetable";

export async function applyRederive(
  videoId: string,
  zones: ZoneInput[],
  opts: { markDone?: boolean },
): Promise<AnalysisResult> {
  // The timetable is read-time data, so a re-derive after someone types the
  // bell times in is exactly how attribution gets its primary rule.
  const video = await getVideo(videoId);
  const period = video ? await periodOffsetsFor(video, await schoolTimezone()) : null;
  const result = await mlRederive(videoId, zones, period);
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
