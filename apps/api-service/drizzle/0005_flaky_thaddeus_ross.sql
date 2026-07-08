-- These columns already exist on some databases (added out-of-band during an
-- earlier board-interaction port), so ADD COLUMN IF NOT EXISTS keeps the
-- migration idempotent across both fresh and previously-patched databases.
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "teacher_pointing_ms" bigint;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "teacher_writing_ms" bigint;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "teacher_board_near_ms" bigint;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN IF NOT EXISTS "board_interactions" jsonb DEFAULT '[]'::jsonb NOT NULL;
