-- Lesson details: the timetable facts a person types in (docs/teacher-measurements.md, P1-P5).
--
-- recording_started_at is the one that makes the rest work. Every measurement
-- compares a video offset (ms from the first frame) against a wall-clock time
-- from the timetable, and those two do not subtract without knowing when the
-- recording itself began. It is read from the container's creation_time during
-- the probe step and stays editable for files that carry none.
--
-- scheduled_start / scheduled_end are `time`, not timestamptz: a period is a
-- fact about the school day rather than an instant. Paired with lesson_date and
-- the school timezone in app_settings, which needs no migration because that
-- table is key-value.
--
-- Every column is nullable. An upload is never blocked on them, and a
-- measurement whose input is absent reports "Not Observed" rather than guessing.
--
-- The video_analytics statements below are re-issued from 0010, which used
-- hand-written `IF NOT EXISTS` that drizzle-kit's snapshot never recorded. They
-- are no-ops on any database that ran 0010; IF NOT EXISTS keeps them harmless
-- on a fresh one too, where 0010 has just created them.
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "teacher_pointing_ms" bigint;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "teacher_writing_ms" bigint;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "pointing_intervals" jsonb DEFAULT '[]'::jsonb NOT NULL;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "writing_intervals" jsonb DEFAULT '[]'::jsonb NOT NULL;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "recording_started_at" timestamp with time zone;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "lesson_date" date;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "period" text;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "scheduled_start" time;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "scheduled_end" time;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "subject" text;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "year_group" text;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "room_type" text;--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN IF NOT EXISTS "has_following_period" boolean;
