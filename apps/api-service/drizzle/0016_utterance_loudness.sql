-- R17 (raised-voice events) is loudness, not words: transcription returns no
-- volume, so it is a separate pass over the extracted FLAC (lib/loudness.ts)
-- run by the audio job after the sentences are cut. Stored per sentence so the
-- event rule — her own speech against her own rolling baseline — is read-time
-- arithmetic like every other voice number, and a changed threshold never
-- re-reads 45 minutes of audio. dBFS; null when the pass has not run.
ALTER TABLE "utterances" ADD COLUMN IF NOT EXISTS "rms_db" real;--> statement-breakpoint
ALTER TABLE "utterances" ADD COLUMN IF NOT EXISTS "peak_db" real;
