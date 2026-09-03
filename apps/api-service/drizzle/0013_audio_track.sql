-- The audio half of the pipeline (docs/teacher-measurements.md, Groups C and D).
--
-- `utterances` stores one row per turn of speech plus what the labelling pass
-- made of it. The unit is the utterance, not the KPI, for the same reason
-- detection_events stores boxes rather than board-minutes: every number in
-- Groups C and D is arithmetic over these rows, so changing a definition is a
-- re-derive rather than paying a transcription API twice.
--
-- The videos.audio_* columns are separate from status/progress on purpose. The
-- two halves run as independent jobs against different services, and one status
-- column with two writers gives the dashboard a badge that flickers between
-- them. Video owns `status`; audio reports in its own lane.
CREATE TABLE IF NOT EXISTS "utterances" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"video_id" uuid NOT NULL,
	"idx" integer NOT NULL,
	"speaker" text NOT NULL,
	"is_teacher" boolean,
	"start_ms" bigint NOT NULL,
	"end_ms" bigint NOT NULL,
	"text" text NOT NULL,
	"confidence" real,
	"language" text,
	"intent" text,
	"attention_cue" boolean,
	"sets_task" boolean
);
--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "audio_status" text;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "audio_error" text;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "audio_path" text;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "transcript_id" text;--> statement-breakpoint
DO $$ BEGIN
	ALTER TABLE "utterances" ADD CONSTRAINT "utterances_video_id_videos_id_fk"
		FOREIGN KEY ("video_id") REFERENCES "public"."videos"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
--> statement-breakpoint
-- Every read is "this lesson's turns, in order" — the derivations walk them as
-- a sequence, and a lesson is thousands of rows once the archive fills up.
CREATE UNIQUE INDEX IF NOT EXISTS "utterances_video_idx" ON "utterances" ("video_id", "idx");
