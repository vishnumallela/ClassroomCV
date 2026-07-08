-- Shrink the hot tier. A 1-hour lesson is ~540k detection_events rows; at fleet
-- scale (80 cams x 8h) that is ~345M rows/day, so holding raw rows for 7 days is
-- untenable. The permanent tiers already preserve everything the dashboard reads
-- (overlay polyline + keyframes in tracks.meta; events / tracks / video_analytics;
-- the track_minute continuous aggregate), so raw rows are only needed for cheap
-- /rederive (re-run merge/roles/events on edited zones without re-running YOLO)
-- and audit.
--
-- VOD analysis writes each video's rows once and never appends, so the chunk is
-- static almost immediately: compress after 1 HOUR (was 24h) for a 3-5x shrink
-- within the hour, and drop after 2 DAYS (was 7) to reclaim the space. 2 days
-- stays comfortably above the track_minute cagg's ~1h refresh lag, so aggregates
-- always materialize before raw rows are dropped. Tune lower with
-- remove_retention_policy + add_retention_policy; going below ~1 day risks the
-- cagg. For per-video immediate drop with cheap rederive intact, persist the
-- ~42-row per-track appearance summary separately (see docs plan M4).

SELECT remove_compression_policy('detection_events', if_exists => true);
--> statement-breakpoint
SELECT add_compression_policy('detection_events', INTERVAL '1 hour');
--> statement-breakpoint
SELECT remove_retention_policy('detection_events', if_exists => true);
--> statement-breakpoint
SELECT add_retention_policy('detection_events', INTERVAL '2 days');
