-- Pointing / writing KPIs.
--
-- The detector has emitted `pointing` (class 3) and `writing` (class 4) since
-- the RF-DETR migration, but nothing consumed them: derive() only ever saw the
-- teacher's own boxes, so the two classes were computed on every frame and
-- discarded. These columns are where they land.
--
-- NULLABLE, like teacher_board_ms and for the same reason. NULL = the input was
-- not available (a /rederive replays teacher-only rows from detection_events
-- and cannot see the action classes at all); 0 = measured, and she did not do
-- it. A NOT NULL DEFAULT 0 here would silently convert every pre-existing row,
-- and every rederive, into a confident claim of zero.
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "teacher_pointing_ms" bigint;
--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "teacher_writing_ms" bigint;
--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "pointing_intervals" jsonb DEFAULT '[]'::jsonb NOT NULL;
--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "writing_intervals" jsonb DEFAULT '[]'::jsonb NOT NULL;
