/** Shared wording for the numbers, so every card says the same thing the same way. */

/** "4.1 min late" / "2 min early" / "on time". Positive is late. */
export function againstBell(
  minutes: number | null | undefined,
  late = "late",
  early = "early",
  onTime = "on time",
): string {
  if (minutes === null || minutes === undefined) return "—";
  if (Math.abs(minutes) < 0.5) return onTime;
  const rounded = Math.abs(Math.round(minutes * 10) / 10);
  return `${rounded} min ${minutes > 0 ? late : early}`;
}

/** Wall-clock "HH:MM" for a video offset, when the recording is anchored. */
export function clockAt(
  recordingStartedAt: string | null,
  ms: number | null | undefined,
  timezone: string,
): string | null {
  if (!recordingStartedAt || ms === null || ms === undefined) return null;
  return new Date(new Date(recordingStartedAt).getTime() + ms).toLocaleTimeString("en-GB", {
    timeZone: timezone,
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function percent(share: number | null | undefined): string {
  return share === null || share === undefined ? "—" : `${Math.round(share * 100)}%`;
}
