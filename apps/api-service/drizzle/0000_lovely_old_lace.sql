CREATE TABLE "detection_events" (
	"video_ts_ms" bigint NOT NULL,
	"video_id" uuid NOT NULL,
	"track_no" integer NOT NULL,
	"bbox" jsonb NOT NULL,
	"confidence" real NOT NULL,
	"meta" jsonb
);
--> statement-breakpoint
CREATE TABLE "events" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"video_id" uuid NOT NULL,
	"track_no" integer,
	"kind" text NOT NULL,
	"video_ts_ms" bigint NOT NULL
);
--> statement-breakpoint
CREATE TABLE "tracks" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"video_id" uuid NOT NULL,
	"track_no" integer NOT NULL,
	"role" text DEFAULT 'unknown' NOT NULL,
	"role_confidence" real,
	"first_ms" bigint NOT NULL,
	"last_ms" bigint NOT NULL,
	"meta" jsonb
);
--> statement-breakpoint
CREATE TABLE "video_analytics" (
	"video_id" uuid PRIMARY KEY NOT NULL,
	"teacher_present_ms" bigint DEFAULT 0 NOT NULL,
	"teacher_board_ms" bigint,
	"entries" integer DEFAULT 0 NOT NULL,
	"exits" integer DEFAULT 0 NOT NULL,
	"avg_students" real,
	"max_students" integer,
	"presence_intervals" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"board_intervals" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"entry_exit" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"occupancy" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"computed_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "videos" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" text NOT NULL,
	"original_filename" text NOT NULL,
	"file_path" text NOT NULL,
	"duration_ms" bigint,
	"fps" real,
	"width" integer,
	"height" integer,
	"status" text DEFAULT 'queued' NOT NULL,
	"progress" real DEFAULT 0 NOT NULL,
	"error" text,
	"workflow_run_id" text,
	"thumbnail_path" text,
	"uploaded_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "zones" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"video_id" uuid NOT NULL,
	"kind" text NOT NULL,
	"polygon" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
ALTER TABLE "events" ADD CONSTRAINT "events_video_id_videos_id_fk" FOREIGN KEY ("video_id") REFERENCES "public"."videos"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tracks" ADD CONSTRAINT "tracks_video_id_videos_id_fk" FOREIGN KEY ("video_id") REFERENCES "public"."videos"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "video_analytics" ADD CONSTRAINT "video_analytics_video_id_videos_id_fk" FOREIGN KEY ("video_id") REFERENCES "public"."videos"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "zones" ADD CONSTRAINT "zones_video_id_videos_id_fk" FOREIGN KEY ("video_id") REFERENCES "public"."videos"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "events_video_idx" ON "events" ("video_id","video_ts_ms");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "tracks_video_idx" ON "tracks" ("video_id");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "zones_video_idx" ON "zones" ("video_id");