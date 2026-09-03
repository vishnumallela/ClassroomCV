import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";

/**
 * Pulling a lesson's audio out of its recording.
 *
 * One extraction serves two consumers: the transcription API (Groups C and D of
 * docs/teacher-measurements.md) and the loudness pass behind R17. So the output
 * is lossless — a lossy codec would be arguing with an RMS measurement — and it
 * is kept on disk rather than re-derived, because re-decoding a 2 GB mp4 to
 * retry one threshold is pure waste.
 */

export interface AudioProbe {
  hasAudio: boolean;
  sampleRate: number | null;
  channels: number | null;
  codec: string | null;
  /**
   * Set when the file could not be read at all — a missing object, a bad URL,
   * a corrupt container.
   *
   * Kept distinct from `hasAudio: false` on purpose. "This recording has no
   * audio track" and "this recording could not be opened" lead to different
   * actions, and reporting the first when the second is true tells someone to
   * go re-record a lesson whose audio was fine all along.
   */
  unreadable: string | null;
}

export interface ExtractedAudio {
  path: string;
  bytes: number;
  sampleRate: number;
}

/**
 * The ceiling worth extracting at. Speech recognition runs on 16 kHz; anything
 * above is discarded downstream, so asking for more only inflates the upload.
 */
const ASR_SAMPLE_RATE = 16_000;

async function run(cmd: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  return { code: await proc.exited, stdout, stderr };
}

/** What the recording's audio track actually is — or that there isn't one. */
export async function probeAudio(source: string): Promise<AudioProbe> {
  const { code, stdout, stderr } = await run([
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "a:0",
    "-show_entries",
    "stream=codec_name,sample_rate,channels",
    "-of",
    "json",
    source,
  ]);
  if (code !== 0) {
    return {
      hasAudio: false,
      sampleRate: null,
      channels: null,
      codec: null,
      unreadable: stderr.trim().slice(-200) || `ffprobe exited ${code}`,
    };
  }

  const parsed = JSON.parse(stdout) as {
    streams?: { codec_name?: string; sample_rate?: string; channels?: number }[];
  };
  const stream = parsed.streams?.[0];
  if (!stream) {
    return { hasAudio: false, sampleRate: null, channels: null, codec: null, unreadable: null };
  }

  const sampleRate = Number(stream.sample_rate);
  return {
    hasAudio: true,
    sampleRate: Number.isFinite(sampleRate) && sampleRate > 0 ? sampleRate : null,
    channels: stream.channels ?? null,
    codec: stream.codec_name ?? null,
    unreadable: null,
  };
}

/**
 * Extract the lesson's audio as mono FLAC.
 *
 * Three decisions, each of which would quietly corrupt a measurement if made
 * the other way:
 *
 * - **No `-ss`, no `-t`.** The audio must keep the video's t=0, or every
 *   transcript timestamp is offset against the detection timeline it has to be
 *   read alongside, and the whole point of a shared clock is lost.
 * - **Never upsample.** Most of this archive is 8 kHz; resampling it to 16 kHz
 *   invents no information, doubles the upload, and makes a telephone-bandwidth
 *   recording look like a good one to anyone reading the file later.
 * - **FLAC, not AAC.** Lossless, so the same file can back the loudness
 *   measurement without a codec's decisions leaking into an RMS figure.
 */
export async function extractAudio(source: string, outPath: string): Promise<ExtractedAudio> {
  const probe = await probeAudio(source);
  if (probe.unreadable) throw new Error(`could not read the recording: ${probe.unreadable}`);
  if (!probe.hasAudio) throw new Error("recording has no audio stream");

  const rate = Math.min(probe.sampleRate ?? ASR_SAMPLE_RATE, ASR_SAMPLE_RATE);

  await mkdir(dirname(outPath), { recursive: true });
  const { code, stderr } = await run([
    "ffmpeg",
    "-y",
    "-i",
    source,
    "-vn",
    "-ac",
    "1",
    "-ar",
    String(rate),
    "-c:a",
    "flac",
    outPath,
  ]);
  if (code !== 0) throw new Error(`ffmpeg exited ${code}: ${stderr.slice(-300)}`);

  const bytes = Bun.file(outPath).size;
  if (bytes === 0) throw new Error("ffmpeg produced an empty audio file");
  return { path: outPath, bytes, sampleRate: rate };
}

/**
 * How much a recording's bandwidth limits what can be read from it.
 *
 * Speech recognition leans on energy between 4 and 8 kHz to tell fricatives
 * apart — /s/ from /f/ from /sh/ — and an 8 kHz recording has none of it, since
 * sampling at 8 kHz caps the signal at 4 kHz. That is not a tuning problem: the
 * information was discarded at the microphone. Reporting it lets a poor
 * transcript be attributed to the recording rather than blamed on the model.
 */
export function audioQualityNote(sampleRate: number | null): string | null {
  if (sampleRate === null) return null;
  if (sampleRate >= 16_000) return null;
  return (
    `Recorded at ${sampleRate} Hz, which caps the audio at ${sampleRate / 2000} kHz and ` +
    "removes the band that separates consonants like s, f and sh. Expect reduced " +
    "transcription accuracy; re-record at 16 kHz or higher to fix it at the source."
  );
}
