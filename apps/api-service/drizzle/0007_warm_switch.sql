CREATE TABLE "classroom_zones" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"classroom_id" uuid NOT NULL,
	"kind" text NOT NULL,
	"polygon" jsonb NOT NULL,
	"meta" jsonb,
	"created_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "classrooms" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" text NOT NULL,
	"location" text,
	"description" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "videos" ADD COLUMN "classroom_id" uuid;--> statement-breakpoint
ALTER TABLE "classroom_zones" ADD CONSTRAINT "classroom_zones_classroom_id_classrooms_id_fk" FOREIGN KEY ("classroom_id") REFERENCES "public"."classrooms"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "videos" ADD CONSTRAINT "videos_classroom_id_classrooms_id_fk" FOREIGN KEY ("classroom_id") REFERENCES "public"."classrooms"("id") ON DELETE restrict ON UPDATE no action;--> statement-breakpoint
-- Hand-written backfill: videos uploaded before classrooms existed land in an
-- auto-created "General" classroom so they stay visible in the new UI.
INSERT INTO "classrooms" ("name", "description")
SELECT 'General', 'Auto-created for lessons uploaded before classrooms existed'
WHERE EXISTS (SELECT 1 FROM "videos" WHERE "classroom_id" IS NULL);--> statement-breakpoint
UPDATE "videos" SET "classroom_id" = (
  SELECT "id" FROM "classrooms" WHERE "name" = 'General' ORDER BY "created_at" LIMIT 1
) WHERE "classroom_id" IS NULL;