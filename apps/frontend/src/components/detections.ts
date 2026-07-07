import type { RouterOutputs } from "@classroom/api-contracts";
import { orpcClient } from "@/lib/orpc";

export type DetectionData = RouterOutputs["videos"]["detections"];
export type DetectionFrame = DetectionData["frames"][number];

export const ROLE_COLORS = {
  teacher: "#10b981",
  student: "#60a5fa",
  unknown: "#a1a1aa",
} as const;

export function colorFor(role: string): string {
  if (role === "teacher") return ROLE_COLORS.teacher;
  if (role === "student") return ROLE_COLORS.student;
  return ROLE_COLORS.unknown;
}

export function roleLabel(role: string): string {
  if (role === "teacher") return "Teacher";
  if (role === "student") return "Student";
  return "Person";
}

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

export function countRoles(roles: Record<string, string>) {
  let teacher = 0;
  let student = 0;
  for (const role of Object.values(roles)) {
    if (role === "teacher") teacher++;
    else if (role === "student") student++;
  }
  return { teacher, student };
}
