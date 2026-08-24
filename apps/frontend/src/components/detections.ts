import type { RouterOutputs } from "@classroom/api-contracts";
import { orpcClient } from "@/lib/orpc";

export type DetectionData = RouterOutputs["videos"]["detections"];
export type DetectionFrame = DetectionData["frames"][number];

// The teacher is the only thing detected, stored and drawn. Students are never
// detected by the model, never written to the database, and never rendered;
// the board and door are zones rather than per-frame boxes.
export const TEACHER_COLOR = "#10b981";

export function fetchDetections(videoId: string, fps = 5): Promise<DetectionData> {
  return orpcClient.videos.detections({ id: videoId, fps });
}

export function findFrameIndex(frames: DetectionFrame[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid]!.tsMs <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

