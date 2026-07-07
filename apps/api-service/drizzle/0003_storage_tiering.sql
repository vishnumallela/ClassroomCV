-- Custom SQL migration file, put your code below! --
-- Re-key detection_events from per-video-relative video_ts_ms (chunks never
-- age, so compression/retention/cagg policies can never fire) to wall-clock
-- ts. TimescaleDB cannot swap the partitioning dimension of an existing
-- hypertable, so the table is rebuilt: rename old, create new keyed on ts,
-- copy rows (ts filled by DEFAULT now()), drop old. Column names/types stay
-- identical so the ML service COPY and the api reads keep working.
ALTER TABLE "detection_events" RENAME TO "detection_events_rekey_old";
--> statement-breakpoint
CREATE TABLE "detection_events" (
	"video_ts_ms" bigint NOT NULL,
	"video_id" uuid NOT NULL,
	"track_no" integer NOT NULL,
	"bbox" jsonb NOT NULL,
	"confidence" real NOT NULL,
	"meta" jsonb,
	"ts" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
SELECT create_hypertable('detection_events','ts',
       chunk_time_interval => INTERVAL '1 hour');
--> statement-breakpoint
INSERT INTO detection_events (video_ts_ms, video_id, track_no, bbox, confidence, meta)
SELECT video_ts_ms, video_id, track_no, bbox, confidence, meta
FROM detection_events_rekey_old;
--> statement-breakpoint
DROP TABLE detection_events_rekey_old;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS detection_events_video_idx
       ON detection_events (video_id, video_ts_ms);
--> statement-breakpoint
ALTER TABLE detection_events SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'video_id',
  timescaledb.compress_orderby   = 'video_ts_ms, track_no'
);
--> statement-breakpoint
SELECT add_compression_policy('detection_events', INTERVAL '24 hours');
--> statement-breakpoint
SELECT add_retention_policy('detection_events', INTERVAL '7 days');
--> statement-breakpoint
CREATE MATERIALIZED VIEW track_minute
WITH (timescaledb.continuous) AS
SELECT video_id,
       time_bucket(INTERVAL '1 minute', ts) AS bucket,
       track_no,
       count(*) AS n
FROM detection_events
GROUP BY 1, 2, 3
WITH NO DATA;
--> statement-breakpoint
CREATE VIEW occupancy_minute AS
SELECT video_id, bucket, count(*) AS bodies
FROM track_minute
GROUP BY 1, 2;
--> statement-breakpoint
SELECT add_continuous_aggregate_policy('track_minute',
  start_offset      => INTERVAL '1 hour',
  end_offset        => INTERVAL '2 minutes',
  schedule_interval => INTERVAL '1 minute');
