export interface ProbeResult {
  durationMs: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  /**
   * Wall-clock instant the recording started, from the container's
   * creation_time tag. This is the hinge every punctuality measurement turns
   * on: detections are offsets in ms from the first frame, the timetable is a
   * clock time, and the two do not subtract without it. 36 of 39 sample
   * recordings carry the tag; null means someone has to type it in.
   */
  recordingStartedAt: Date | null;
}

interface FfprobeStream {
  codec_type?: string;
  avg_frame_rate?: string;
  r_frame_rate?: string;
  width?: number;
  height?: number;
  tags?: { creation_time?: string };
}

interface FfprobeOutput {
  streams?: FfprobeStream[];
  format?: { duration?: string; tags?: { creation_time?: string } };
}

/**
 * Reject a creation_time that cannot be a lesson. Some encoders write a zero
 * epoch or a placeholder year, and a bogus instant is worse than none here:
 * it silently produces a confident "47 minutes late".
 */
function parseCreationTime(value: string | undefined): Date | null {
  if (!value) return null;
  const at = new Date(value);
  if (Number.isNaN(at.getTime())) return null;
  const year = at.getUTCFullYear();
  if (year < 2000 || year > 2100) return null;
  return at;
}

async function run(cmd: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  const code = await proc.exited;
  return { code, stdout, stderr };
}

function parseFrameRate(value: string | undefined): number | null {
  if (!value) return null;
  const parts = value.split("/");
  const num = Number(parts[0]);
  const den = Number(parts[1] ?? "1");
  if (!Number.isFinite(num) || !Number.isFinite(den) || den === 0) return null;
  const fps = num / den;
  return Number.isFinite(fps) && fps > 0 ? fps : null;
}

export async function probeVideo(filePath: string): Promise<ProbeResult> {
  const { code, stdout, stderr } = await run([
    "ffprobe",
    "-v",
    "error",
    "-print_format",
    "json",
    "-show_format",
    "-show_streams",
    filePath,
  ]);
  if (code !== 0) throw new Error(`ffprobe exited ${code}: ${stderr.slice(0, 200)}`);

  const parsed = JSON.parse(stdout) as FfprobeOutput;
  const stream = parsed.streams?.find((s) => s.codec_type === "video");
  const durationSec = Number(parsed.format?.duration);
  // MediaRecorder .webm carries no duration header, so this stays null and the
  // real duration is backfilled from the analysis result at ingest time.
  const durationMs =
    Number.isFinite(durationSec) && durationSec > 0 ? Math.round(durationSec * 1000) : null;
  const fps = parseFrameRate(stream?.avg_frame_rate) ?? parseFrameRate(stream?.r_frame_rate);
  // Format tag first: it is the container's own record. The video stream's
  // copy is the fallback, and the two agree wherever both are present.
  const recordingStartedAt =
    parseCreationTime(parsed.format?.tags?.creation_time) ??
    parseCreationTime(stream?.tags?.creation_time);
  return {
    durationMs,
    fps,
    width: stream?.width ?? null,
    height: stream?.height ?? null,
    recordingStartedAt,
  };
}

export async function generateThumbnail(
  filePath: string,
  outPath: string,
  atSeconds: number,
): Promise<boolean> {
  const { code } = await run([
    "ffmpeg",
    "-y",
    "-ss",
    atSeconds.toFixed(3),
    "-i",
    filePath,
    "-frames:v",
    "1",
    "-q:v",
    "4",
    outPath,
  ]);
  return code === 0;
}
