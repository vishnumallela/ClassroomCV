ALTER TABLE "video_analytics" ADD COLUMN "teacher_pointing_ms" bigint;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN "teacher_writing_ms" bigint;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN "teacher_board_near_ms" bigint;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD COLUMN "board_interactions" jsonb DEFAULT '[]'::jsonb NOT NULL;