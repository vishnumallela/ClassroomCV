import { type Job, UnrecoverableError } from "bullmq";
import { dirname, join } from "node:path";
import { getVideo, replaceUtterances, setAudioStatus, type UtteranceInput } from "@api/db/queries";
import { isS3 } from "@api/lib/storage";
import { audioQualityNote, extractAudio, probeAudio } from "@api/lib/audio";
import { logger } from "@api/lib/logger";
import { languageOf, segmentSentences } from "@api/lib/segment";
import { mediaSource } from "@api/jobs/analyze-video";
import type { AudioJobData } from "@api/lib/queue";
import {
  getTranscript,
  MIN_TRANSCRIBE_SAMPLE_RATE,
  submitTranscript,
  TranscribeNotConfiguredError,
} from "@api/lib/transcribe";

/**
 * The audio half of a lesson: extract, transcribe, store the turns.
 *
 * Runs entirely independently of the video job. It needs no GPU, so it must
 * never be gated on the ML service being reachable — a lesson uploaded with the
 * pod off should still come back with its transcript. Both halves are
 * submit-then-poll against remote services, so running them at once costs
 * almost nothing in wall clock.
 *
 * Nothing here writes `videos.status`. That column belongs to the video job;
 * two writers on it produce a status badge that flickers between them.
 */

const POLL_INTERVAL_MS = 10_000;
// Transcription runs far faster than real time, so an hour of polling is a
// hung job rather than a slow one.
const MAX_POLLS = (60 * 60) / 10;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Reasons to stop before spending anything, each a legitimate outcome rather
 * than a failure: the lesson simply has no audio measurements, and every Group
 * A number still stands.
 */
async function skipReason(source: string): Promise<string | null> {
  const probe = await probeAudio(source);
  // Unreadable is not a skip. It means the bytes are missing or corrupt, which
  // is a fault to fix rather than a property of the lesson — and saying "no
  // audio track" here would send someone to re-record a fine recording.
  if (probe.unreadable) throw new Error(`could not read the recording: ${probe.unreadable}`);
  if (!probe.hasAudio) return "This recording has no audio track.";
  if ((probe.sampleRate ?? 0) < MIN_TRANSCRIBE_SAMPLE_RATE) {
    return (
      audioQualityNote(probe.sampleRate) ??
      `Audio is ${probe.sampleRate} Hz, below the ${MIN_TRANSCRIBE_SAMPLE_RATE} Hz minimum.`
    );
  }
  return null;
}

export async function processAudioJob(job: Job<AudioJobData>): Promise<void> {
  const { videoId } = job.data;

  const video = await getVideo(videoId);
  if (!video) throw new UnrecoverableError(`video ${videoId} was deleted before audio analysis`);

  // Local copy first. The object store is authoritative for anything a remote
  // GPU pod must fetch, but this job runs here, and lessons uploaded before the
  // 2026-07-22 switch to MinIO have their bytes only on local disk.
  const source = (await Bun.file(video.filePath).exists())
    ? video.filePath
    : isS3
      ? mediaSource(video.filePath)
      : video.filePath;

  try {
    const skip = await skipReason(source);
    if (skip) {
      logger.info({ videoId, skip }, "audio analysis skipped");
      await setAudioStatus(videoId, { audioStatus: "skipped", audioError: skip });
      return;
    }

    // Extract beside the video, so deleting the lesson takes the audio with it.
    await setAudioStatus(videoId, { audioStatus: "extracting", audioError: null });
    const out = join(dirname(video.filePath), "audio.flac");
    const audio = await extractAudio(source, out);
    logger.info(
      { videoId, bytes: audio.bytes, sampleRate: audio.sampleRate },
      "lesson audio extracted",
    );
    await setAudioStatus(videoId, { audioPath: audio.path });

    // Reuse an in-flight transcript across retries. Transcription is billed per
    // hour of audio, so resubmitting the same lesson after a transient poll
    // failure is paying twice for one answer.
    let transcriptId = video.transcriptId;
    if (!transcriptId) {
      await setAudioStatus(videoId, { audioStatus: "transcribing" });
      transcriptId = await submitTranscript(audio.path, {
        keyTerms: [video.subject, video.yearGroup].filter((v): v is string => Boolean(v)),
      });
      await setAudioStatus(videoId, { transcriptId });
    }

    const result = await pollTranscript(transcriptId, videoId);

    // Sentences, not diarizer turns: a turn runs until somebody else speaks,
    // which on a real lesson was 6.5 minutes and 3,650 characters in one row.
    // Cut from the words (lib/segment.ts) when the provider returned them;
    // the turns stand in only when it did not.
    const sentences =
      result.words.length > 0
        ? segmentSentences(result.words)
        : result.utterances.map((u) => ({
            speaker: u.speaker,
            start: u.start,
            end: u.end,
            text: u.text,
            confidence: u.confidence,
            language: languageOf(u.text),
          }));
    const rows: UtteranceInput[] = sentences.map((u, idx) => ({
      idx,
      speaker: u.speaker,
      // Left null on purpose. The diarizer's label is one opinion; the video's
      // presence intervals are another, and which voice is hers is decided at
      // read time from both (lib/voice.ts), so there is no second copy to drift.
      isTeacher: null,
      startMs: u.start,
      endMs: u.end,
      text: u.text,
      confidence: u.confidence ?? null,
      language: u.language,
    }));
    await replaceUtterances(videoId, rows);

    await setAudioStatus(videoId, {
      audioStatus: rows.length > 0 ? "done" : "empty",
      audioError: rows.length > 0 ? null : "Transcription returned no speech.",
    });
    logger.info(
      { videoId, utterances: rows.length, languages: result.detectedLanguages },
      "audio analysis complete",
    );
  } catch (err) {
    // A missing API key is a configuration gap, not a lesson that failed:
    // retrying it five times changes nothing and buries the real message.
    if (err instanceof TranscribeNotConfiguredError) {
      await setAudioStatus(videoId, { audioStatus: "skipped", audioError: err.message });
      logger.warn({ videoId }, "audio analysis skipped: no transcription key configured");
      return;
    }
    const terminal =
      err instanceof UnrecoverableError || job.attemptsMade + 1 >= (job.opts.attempts ?? 1);
    if (terminal) {
      await setAudioStatus(videoId, {
        audioStatus: "failed",
        audioError: err instanceof Error ? err.message : String(err),
      }).catch(() => undefined);
    }
    throw err;
  }
}

async function pollTranscript(transcriptId: string, videoId: string) {
  for (let attempt = 0; attempt < MAX_POLLS; attempt++) {
    const result = await getTranscript(transcriptId);
    if (result.status === "completed") return result;
    if (result.status === "error") {
      throw new UnrecoverableError(`transcription failed: ${result.error ?? "unknown error"}`);
    }
    logger.debug({ videoId, transcriptId, status: result.status }, "waiting on transcript");
    await sleep(POLL_INTERVAL_MS);
  }
  throw new UnrecoverableError("transcription did not complete within an hour");
}
