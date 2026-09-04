import { spawn } from "node:child_process";

/**
 * R17 — raised-voice events — is loudness, not words. Transcription returns no
 * volume, so this is the one voice measurement that reads the waveform: the
 * extracted 16 kHz FLAC, in fixed windows, through ffmpeg's `astats`. The
 * per-window RMS is folded onto each sentence (mean and peak) and stored; the
 * event rule then runs at read time over her sentences against her own
 * baseline, so a changed threshold never re-reads the audio.
 *
 * "Her own baseline" is the point: a classroom microphone's absolute level
 * says more about where the mic hangs than about the teacher, and a teacher
 * who speaks loudly all lesson has not raised her voice.
 */

export interface LoudnessWindow {
  /** Window start, seconds into the recording. */
  t: number;
  /** RMS level of the window in dBFS (negative; 0 is full scale). */
  rms: number;
}

export const WINDOW_SEC = 0.5;
/** A sentence this far above her baseline is a raised voice... */
export const RAISED_DB = 6;
/** ...when it lasts at least this long, so a single shout of a name is not an event. */
export const RAISED_MIN_MS = 1_500;
/** Events closer than this are one episode. */
export const RAISED_MERGE_MS = 5_000;
/** Windows quieter than this are treated as silence and do not weigh a sentence. */
export const SILENCE_DB = -60;

/** Parse ffmpeg `ametadata=print` output into windows. */
export function parseAstats(output: string): LoudnessWindow[] {
  const out: LoudnessWindow[] = [];
  let t: number | null = null;
  for (const line of output.split("\n")) {
    const frame = /pts_time:([0-9.]+)/.exec(line);
    if (frame) {
      t = Number(frame[1]);
      continue;
    }
    const rms = /lavfi\.astats\.Overall\.RMS_level=(-?[0-9.]+|-inf)/.exec(line);
    if (rms && t !== null) {
      const v = rms[1] === "-inf" ? -120 : Number(rms[1]);
      if (Number.isFinite(v)) out.push({ t, rms: v });
      t = null;
    }
  }
  return out;
}

export async function measureLoudness(
  flacPath: string,
  ffmpeg = "ffmpeg",
  windowSec = WINDOW_SEC,
): Promise<LoudnessWindow[]> {
  // 16 kHz mono is what lib/audio.ts extracts; a different rate only changes
  // the window length, not the measurement.
  const samples = Math.round(16_000 * windowSec);
  const args = [
    "-v",
    "error",
    "-i",
    flacPath,
    "-af",
    `asetnsamples=n=${samples}:p=0,astats=metadata=1:reset=1,` +
      "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
    "-f",
    "null",
    "-",
  ];
  const output = await new Promise<string>((resolve, reject) => {
    const child = spawn(ffmpeg, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
    child.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve(stdout);
      else reject(new Error(`ffmpeg loudness pass exited ${code}: ${stderr.slice(0, 300)}`));
    });
  });
  return parseAstats(output);
}

/**
 * A sentence's loudness from the windows it covers: the mean of the voiced
 * windows (as a power average, so one loud window is not diluted by ten quiet
 * ones the way a dB average would) and the peak window.
 */
export function sentenceLoudness(
  windows: LoudnessWindow[],
  startMs: number,
  endMs: number,
  windowSec = WINDOW_SEC,
): { rmsDb: number | null; peakDb: number | null } {
  const from = startMs / 1000;
  const to = endMs / 1000;
  let power = 0;
  let n = 0;
  let peak = -Infinity;
  for (const w of windows) {
    if (w.t + windowSec <= from || w.t >= to) continue;
    if (w.rms < SILENCE_DB) continue;
    power += 10 ** (w.rms / 10);
    n++;
    if (w.rms > peak) peak = w.rms;
  }
  if (n === 0) return { rmsDb: null, peakDb: null };
  return {
    rmsDb: Math.round(10 * Math.log10(power / n) * 10) / 10,
    peakDb: Math.round(peak * 10) / 10,
  };
}

export interface RaisedVoiceEvent {
  startMs: number;
  endMs: number;
  /** How far above her baseline the loudest sentence in the episode sat. */
  dbAbove: number;
}

/**
 * Her sentences RAISED_DB above her own baseline — the median of her
 * sentences' RMS — for at least RAISED_MIN_MS, merged into episodes.
 */
export function raisedVoiceEvents(
  sentences: { startMs: number; endMs: number; rmsDb: number | null }[],
): { events: RaisedVoiceEvent[]; baselineDb: number | null } {
  const levels = sentences
    .map((s) => s.rmsDb)
    .filter((v): v is number => v !== null)
    .sort((a, b) => a - b);
  if (levels.length < 10) return { events: [], baselineDb: null };
  const baseline = levels[Math.floor(levels.length / 2)] ?? null;
  if (baseline === null) return { events: [], baselineDb: null };

  const events: RaisedVoiceEvent[] = [];
  for (const s of [...sentences].sort((a, b) => a.startMs - b.startMs)) {
    if (s.rmsDb === null || s.endMs - s.startMs < RAISED_MIN_MS) continue;
    const above = s.rmsDb - baseline;
    if (above < RAISED_DB) continue;
    const last = events[events.length - 1];
    if (last && s.startMs - last.endMs <= RAISED_MERGE_MS) {
      last.endMs = Math.max(last.endMs, s.endMs);
      last.dbAbove = Math.max(last.dbAbove, Math.round(above * 10) / 10);
    } else {
      events.push({ startMs: s.startMs, endMs: s.endMs, dbAbove: Math.round(above * 10) / 10 });
    }
  }
  return { events, baselineDb: Math.round(baseline * 10) / 10 };
}
