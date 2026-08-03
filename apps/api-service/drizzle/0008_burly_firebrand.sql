ALTER TABLE "video_analytics" ALTER COLUMN "heatmap" SET DEFAULT '{"grid_w":0,"grid_h":0,"teacher":[]}'::jsonb;--> statement-breakpoint
ALTER TABLE "video_analytics" DROP COLUMN "avg_students";--> statement-breakpoint
ALTER TABLE "video_analytics" DROP COLUMN "max_students";--> statement-breakpoint
ALTER TABLE "video_analytics" DROP COLUMN "occupancy";