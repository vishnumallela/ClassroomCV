export interface ProbeResult {
  durationMs: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
}

interface FfprobeStream {
  codec_type?: string;
  avg_frame_rate?: string;
  r_frame_rate?: string;
  width?: number;
  height?: number;
}

interface FfprobeOutput {
  streams?: FfprobeStream[];
  format?: { duration?: string };
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
  return {
    durationMs,
    fps,
    width: stream?.width ?? null,
    height: stream?.height ?? null,
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
