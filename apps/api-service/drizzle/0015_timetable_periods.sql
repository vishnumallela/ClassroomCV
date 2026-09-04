-- The classroom's week, typed once (docs/lesson-coverage-plan.md, Phase D).
--
-- Every punctuality number is a subtraction between a video offset and a bell
-- time, and until now the bell times lived on the video row — a fact about the
-- classroom's week stored as a fact about a file, typed again for every upload.
-- This table holds one row per teaching period per ISO weekday (1 = Monday).
-- Breaks are the gaps between rows, so "is there a period straight after this
-- one" and "when did the previous period end" are derived rather than typed;
-- the second is what lets the previous period's teacher be measured against
-- HER bell (09:25 at this school, with a 25-minute break before period 3)
-- instead of against this period's start.
--
-- The per-video columns stay: as an override when someone corrects one lesson,
-- and as the only source for lessons recorded before this table existed.
CREATE TABLE IF NOT EXISTS "timetable_periods" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "classroom_id" uuid NOT NULL,
  "weekday" smallint NOT NULL,
  "slot" smallint NOT NULL,
  "label" text NOT NULL,
  "scheduled_start" time NOT NULL,
  "scheduled_end" time NOT NULL,
  "subject" text,
  "teacher" text,
  "year_group" text,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL,
  CONSTRAINT "timetable_periods_weekday_check" CHECK ("weekday" BETWEEN 1 AND 7),
  CONSTRAINT "timetable_periods_window_check" CHECK ("scheduled_end" > "scheduled_start")
);--> statement-breakpoint
ALTER TABLE "timetable_periods" ADD CONSTRAINT "timetable_periods_classroom_id_classrooms_id_fk" FOREIGN KEY ("classroom_id") REFERENCES "public"."classrooms"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "timetable_periods_classroom_weekday_slot" ON "timetable_periods" USING btree ("classroom_id","weekday","slot");
