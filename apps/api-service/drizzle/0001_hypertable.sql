-- Custom SQL migration file, put your code below! --
CREATE EXTENSION IF NOT EXISTS timescaledb;
--> statement-breakpoint
SELECT create_hypertable('detection_events','video_ts_ms',
       chunk_time_interval => 3600000, if_not_exists => TRUE);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS detection_events_video_idx
       ON detection_events (video_id, video_ts_ms);
