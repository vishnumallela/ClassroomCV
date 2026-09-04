/**
 * The system, pin to pin — the content of the "How it works" tab.
 *
 * Every value here is the constant the code actually runs (named in
 * `config` with the file that owns it), and every step says what goes in,
 * what happens, what comes out, where it goes next, and who is responsible.
 * When a number changes in the code, change it here in the same commit.
 */

export interface Step {
  name: string;
  owner: string;
  in: string;
  processing: string;
  out: string;
  next: string;
  config?: string[];
}

export interface Section {
  id: string;
  n: number;
  title: string;
  intro: string;
  steps: Step[];
}

export const SECTIONS: Section[] = [
  {
    id: "ingestion",
    n: 1,
    title: "Video input and ingestion",
    intro:
      "A recording enters as one file per period from a fixed classroom camera. Nothing decodes it on the way in: the API streams the bytes to storage, reads the container header, queues two jobs, and the GPU fetches its own copy when it is ready.",
    steps: [
      {
        name: "Source file",
        owner: "School CCTV export",
        in: "One MP4 per period from a fixed camera (H.264 or HEVC, 1920×1080 or 2560×1440, ~25 fps, 45 min ≈ 0.5–1 GB), with a burned-in wall clock in the top-right corner.",
        processing:
          "None. The school's export is usually an ffmpeg re-encode, which strips the container's creation_time, so the burned-in clock is the only time anchor and is typed in (clock OCR is planned).",
        out: "The file, as uploaded from the dashboard into a classroom.",
        next: "Upload.",
        config: [
          "Recording protocol: record on the bell, run past the end bell, upload originals (docs/lesson-coverage-plan.md §4)",
        ],
      },
      {
        name: "Upload",
        owner: "API · apps/api-service/src/server/routes.ts",
        in: "POST /videos?filename=… with the raw file as the body, from the classroom's Lessons page.",
        processing:
          "Inserts the videos row (status queued), seeds the classroom's board/door zone template into the video's zones, then streams the body through a write sink: a local copy under DATA_DIR/videos/<id>/original.mp4 and, with the S3 backend, the same bytes into the MinIO/S3 bucket. Rejects bodies over the size cap. Sets a workflow run id as a fence so a later re-analysis can supersede this run.",
        out: "videos row + stored bytes.",
        next: "Two jobs are enqueued at once: analyze (video queue) and analyze-audio (audio queue). They run against different services, so the lesson's turnaround is the longer of the two, not the sum.",
        config: [
          "API_SERVICE__MAX_UPLOAD_BYTES = 4 GiB",
          "API_SERVICE__STORAGE_BACKEND = s3 (MinIO on-prem) or local",
          "BullMQ job options: 5 attempts, exponential backoff from 2 s",
        ],
      },
      {
        name: "Probe and thumbnail",
        owner: "API worker · src/jobs/analyze-video.ts probeStep",
        in: "The stored file, reached by a 6-hour presigned URL (S3) or the local path.",
        processing:
          "ffprobe reads the header only: duration, fps, width, height and the creation_time tag. A valid creation_time seeds recordingStartedAt and lessonDate on the row, never overwriting a typed value. ffmpeg seeks one frame at 10% of the duration for the thumbnail, uploaded to the object store. Status moves queued → probing → analyzing.",
        out: "videos.duration_ms / fps / width / height (+ recording_started_at when the tag existed), thumb.jpg.",
        next: "Start analysis on the ML service.",
        config: [
          "Thumbnail mark: 10% of duration",
          "Status polling from the UI every 2 s while not done",
        ],
      },
      {
        name: "Hand-off to the GPU",
        owner: "API worker · startAnalysisStep → ML service POST /analyze",
        in: "videoId, a media source (presigned URL), sample_fps 5, the video's zones, the period as ms offsets (from the timetable), an idempotency key and run tokens.",
        processing:
          "Checks GET /health on the ML service first and refuses to start unless the device is cuda — a mis-scheduled pod would otherwise bill a CPU run. The ML service accepts with 202 and a job id; the worker polls GET /jobs/{id} every 5 s and then fetches /jobs/{id}/result.",
        out: "An ML job running on the pod.",
        next: "Section 2.",
        config: [
          "POLL_INTERVAL_MS 5 000",
          "processing budget 2 h, queue wait up to 24 h",
          "ML URL = https://<podId>-8000.proxy.runpod.net when a pod exists, else API_SERVICE__ML_SERVICE_URL",
        ],
      },
    ],
  },
  {
    id: "video-pipeline",
    n: 2,
    title: "Video processing pipeline",
    intro:
      "One ML job per video, in strict order: fetch, decode and sample, detect in batches, describe, propose zones, track, attribute, derive, assess, persist. Detection is reported as progress 0–0.9 and everything after it as 0.9–1.0.",
    steps: [
      {
        name: "Fetch the media",
        owner: "ML service · app/detector.py _download_to_temp",
        in: "The presigned URL from the API.",
        processing:
          "The URL's host must be on MEDIA_URL_ALLOWLIST (the presigned host must match exactly, since SigV4 signs the Host header). The file is downloaded in 1 MiB chunks into DATA_DIR with HTTP Range resume on failure and a size cap. On a rented pod the URL points back at the on-site MinIO through a reverse SSH tunnel.",
        out: "A local file DATA_DIR/mediacache_*.mp4 (513 MB for the 45-minute lesson, ~3.5 min through the tunnel).",
        next: "Decode.",
        config: [
          "MEDIA_URL_ALLOWLIST = 127.0.0.1:9000",
          "MEDIA_MAX_DOWNLOAD_BYTES 8 GiB",
          "retries 5, timeout 60 s",
        ],
      },
      {
        name: "Decode and sample",
        owner: "ML service · app/detector.py iter_frames",
        in: "The local file.",
        processing:
          "cv2.VideoCapture decodes EVERY frame (H.264 is inter-coded, so no frame can be skipped) and retrieves only every stride-th, where stride = round(native_fps / 5). Each kept frame is stamped ts_ms = frame_index / native_fps × 1000 — the clock every later number lives on. This stage is CPU-bound: on a 45-minute 1080p file the GPU idles at 2–7% while decode runs at 100–280% CPU.",
        out: "(ts_ms, BGR frame) pairs at ~5 fps: 13 498 frames for 45 minutes.",
        next: "Batching.",
        config: [
          "MAX_SAMPLE_FPS 5.0",
          "FALLBACK_NATIVE_FPS 30",
          "duration from frame count / native fps",
        ],
      },
      {
        name: "Batch and infer",
        owner: "ML service · detect_video → _predict_batch",
        in: "Sampled frames.",
        processing:
          "Frames are buffered to the batch size and sent to RF-DETR in one predict call. On CUDA the model is JIT-traced once at that exact batch size in fp16; a ragged last batch is padded by repeating its final frame and the padding dropped from the results. RF-DETR letterboxes each frame to its square resolution internally. Off-GPU (dev on Apple MPS) the batch is 1 and precision fp32.",
        out: "One supervision Detections set per frame: class ids, confidences, xyxy boxes.",
        next: "Conversion to normalised rows.",
        config: [
          "RFDETR_BATCH 16 on CUDA (1 elsewhere)",
          "RFDETR_RESOLUTION 576",
          "DETECT_FLOOR 0.15 — nothing below it leaves the detector",
          "fp16 on CUDA only; TensorRT wired but unverified",
        ],
      },
      {
        name: "Rows and appearance",
        owner: "ML service · _to_detections + app/appearance.py",
        in: "Per-frame detections and the frame's pixels.",
        processing:
          "Boxes are normalised to 0–1 {x, y, w, h}, clamped to the frame. Every teacher-class box with conf ≥ 0.3 also gets an appearance descriptor computed right here, because this is the only moment the pixels exist: a 16-number HSV histogram (8 hue bins weighted by saturation, 4 saturation, 4 value) of the central torso band of the box (15–75% of its height, 20–80% of its width) on a 24-pixel patch.",
        out: "Detection {video_ts_ms, cls, bbox, conf, app} for every box ≥ 0.15.",
        next: "The whole list goes to derive_result once detection finishes.",
        config: [
          "DESCRIBE_MIN_CONF 0.3",
          "H_BINS 8 · S_BINS 4 · V_BINS 4",
          "band 0.15–0.75 × 0.20–0.80 · PATCH 24",
        ],
      },
      {
        name: "Zones: propose, then gate",
        owner: "ML service · app/zones.py",
        in: "All detections plus the zones the API sent.",
        processing:
          "If the room has no board or door zone, one is proposed from the lesson's own screen/door boxes: each edge is the median across the video, accepted when the class was present in enough frames. The proposal is used in this same run and returned to the API, which stores it as auto-placed. Then gate_static drops board/door boxes outside their (configured) zone; the teacher and the action classes are never gated.",
        out: "Zones for this run; a filtered detection list.",
        next: "Tracking.",
        config: [
          "MIN_PRESENCE 0.30 · MIN_SAMPLES 8",
          "AUTO_ACCEPT board 0.5 / door 0.4",
          "GATE_TOLERANCE 0.05",
          "zone_conf 0.5",
        ],
      },
      {
        name: "Derive, assess, persist",
        owner: "ML service · app/jobs.py run_pipeline / derive_result",
        in: "The gated detections, the zones, the period offsets.",
        processing:
          "Tracking (§4), attribution (§4), events and analytics (§5), per-person analytics for every tracked adult, and the quality report run as pure functions over the list — the same code path /rederive uses on stored rows, which is why any later rule change is a cheap replay rather than a GPU re-run. Then the teacher's boxes (≥ 0.4) and her pointing/writing boxes (≥ 0.5) are COPYed straight into TimescaleDB, each with its class and descriptor in meta and the person number as track_no.",
        out: "detection_events rows on the school's Postgres; an AnalysisResult JSON (video meta, tracks with overlays and per-person analytics, events, analytics, data_quality, proposed zones).",
        next: "The API worker ingests the result (§11).",
        config: [
          "teacher_conf 0.4 · action_conf 0.5",
          "COPY over a reverse SSH tunnel to 127.0.0.1:5533 from a pod",
          "progress: detecting 0–0.9, deriving 0.9–1.0",
        ],
      },
    ],
  },
  {
    id: "detection",
    n: 3,
    title: "AI detection",
    intro:
      "One model. A fine-tuned RF-DETR names the teacher directly, so nothing downstream has to infer which person is the adult, and no student is ever a class.",
    steps: [
      {
        name: "RF-DETR (medium), fine-tuned",
        owner: "ML service · app/detector.py",
        in: "A batch of BGR frames.",
        processing:
          "A DETR-family detection transformer (rfdetr 1.9.3, RFDETRMedium, PyTorch 2.12 cu13) trained on this product's five classes. Object queries attend over the whole frame and emit boxes directly, with no proposals and no NMS. The checkpoint is a 255 MB file uploaded to the pod at /workspace/weights/rfdetr-medium.pth.",
        out: "Per frame: N × {class, bbox, conf}.",
        next: "Per-class thresholds are applied by the consumers, so one pass serves all of them.",
        config: [
          "Classes: 0 door · 1 screen (the board) · 2 teacher · 3 pointing · 4 writing",
          "Emitted ≥ 0.15; teacher used ≥ 0.4; zones ≥ 0.5; actions ≥ 0.5; descriptors ≥ 0.3",
          "Eval: 95.5% of scored frames carry a correct teacher box; at most one teacher box per frame on the single-adult held-out room",
        ],
      },
      {
        name: "What each class becomes",
        owner: "ML service · consumers of the detection list",
        in: "The class of a box.",
        processing:
          "teacher → the tracker and everything built on presence. screen/door → zone proposal only, never stored. pointing/writing → attributed to the teacher's box in the same frame (§5) and stored with her boxes so a re-derive keeps them.",
        out: "Stored: her boxes and her gesture boxes. Not stored: screen, door, and anything below its threshold — there is no student box in the database.",
        next: "Tracking (§4).",
        config: [
          "Privacy: no face recognition, no student class; students are neither detected nor drawn",
        ],
      },
      {
        name: "Explicitly not a vision-language model",
        owner: "Design record · docs/rfdetr-migration",
        in: "—",
        processing:
          "The previous pipeline used a Gemini vision vote and a CLIP re-identification encoder to decide who the teacher was. Both went out on 24 August with the fine-tuned detector (−7 500 lines). Every judgement since is either a trained class or an explainable rule, so every number traces to a cause.",
        out: "—",
        next: "—",
      },
    ],
  },
  {
    id: "tracking",
    n: 4,
    title: "Tracking and identity",
    intro:
      "The camera is fixed and single; identity is decided per file. First the motion model follows every teacher-class body into segments, then attribution turns segments into people and names the one this period is about. Cross-file identity is not appearance-based: the timetable and the register do that (§9).",
    steps: [
      {
        name: "Per-instant dedup and assignment",
        owner: "ML service · app/teacher.py _track",
        in: "Teacher boxes (≥ 0.4) grouped by sampled instant.",
        processing:
          "Two boxes at one instant are one body when the smaller sits mostly inside the larger (containment, not IoU, because a half-body box on a full-body box has IoU ≈ 0.4). Each active segment is extended by the nearest reachable box to its PREDICTED position — its last centre carried forward by its last velocity — one box per segment per instant, greedy by distance. Reachability is judged from the last sighting: allowed distance = 0.05 + 0.8 × gap_s. Past a 5-second gap position carries no information.",
        out: "Segments: bodies followed without ambiguity, cut wherever the motion model could not decide.",
        next: "Welding and the noise floor.",
        config: [
          "SAME_BODY_OVERLAP 0.5",
          "JUMP_BASE 0.05 · MAX_SPEED_PER_S 0.8 · FREE_GAP_MS 5 000",
          "PREDICT_MAX_STEP_MS 1 000 · PREDICT_MAX_LEAD_MS 600",
        ],
      },
      {
        name: "Returns, noise, co-presence",
        owner: "ML service · app/teacher.py",
        in: "Leftover boxes and finished segments.",
        processing:
          "A box no active segment claims welds onto a dormant segment only when that return is unambiguous (no other substantial segment was around when it went dormant); otherwise it starts a new segment. Segments shorter than 3 s or 8 boxes are noise — a student the detector called teacher for a few frames — and are never numbered or surfaced. Instants offering two or more boxes are counted; 30 s of them flags the room as multi-adult, which withholds punctuality unless attribution is high.",
        out: "Substantial segments, an interim primary (the biggest), co-presence counts.",
        next: "Attribution.",
        config: ["SEGMENT_MIN_MS 3 000 · SEGMENT_MIN_DETS 8", "CO_PRESENCE_MIN_MS 30 000"],
      },
      {
        name: "Undo swaps at occluded crossings",
        owner: "ML service · app/attribution.py resolve_swaps",
        in: "Substantial segments with descriptors.",
        processing:
          "When two people cross while one is briefly undetected, the motion model hands each lane the other body. At every instant two lanes are within reach of each other, both lanes' appearance windows (6 s before and after) are compared as tracked versus re-paired; the tails are swapped when re-pairing is cheaper by the margin. This is what the change-point split cannot do: black against white scores 0.33 under this descriptor, below what one teacher reaches through an ordinary occlusion.",
        out: "Lanes that each carry one body.",
        next: "Split, merge, link.",
        config: [
          "SWAP_WINDOW_MS 6 000 · SWAP_MARGIN 0.25 · SWAP_MIN_CLEAN 5",
          "measured at the real swap: 0.77 as tracked vs 0.36 re-paired",
        ],
      },
      {
        name: "Split, merge duplicates, link into people",
        owner: "ML service · app/attribution.py",
        in: "Resolved lanes.",
        processing:
          "split_at_changes cuts a lane where its appearance changes and stays changed (30-box windows, threshold 0.42). merge_duplicate_lanes folds a head-only lane into the torso lane it shadows in the same column. link_people joins pieces into people: overlapping pieces only at the same place; adjacent pieces by position with appearance as a tie-breaker; pieces across a long gap by appearance alone, which must clear 0.30. A piece speaks for its appearance only with 8 clean boxes that are also half of it.",
        out: "People: ordered lists of segments.",
        next: "Choosing the period's teacher.",
        config: [
          "SPLIT_WINDOW 30 · SPLIT_THRESHOLD 0.42",
          "SAME_COLUMN 0.8 · SAME_COLUMN_DY 0.20",
          "OVERLAP_TOL_MS 3 000 · SAME_PLACE 0.10 · POS_WEIGHT 0.30 · APP_LINK_MAX 0.30",
          "CLEAN_MIN_H 0.15 · CLEAN_MIN_CONF 0.5 · CLEAN_MIN_COUNT 8 · CLEAN_MIN_SHARE 0.5",
        ],
      },
      {
        name: "Which person is this period's teacher",
        owner: "ML service · app/attribution.py attribute",
        in: "People, the recording length, the period as offsets.",
        processing:
          "A person whose last sighting is followed by another adult's presence for 10 s has handed the room over. If exactly one person stayed and others handed over, she is the teacher: high confidence when she then held the room alone for 60 s or was present twice as long as anyone who left, else medium. With nobody handing over, presence within the period decides, with a 2× lead for high and 1.3× for medium; closer than that is undetermined and every punctuality number is withheld. Each person's departure is the end of her presence run containing the bell, bridging absences under 5 minutes.",
        out: "Attribution {chosen person, confidence, reason, candidates with first/last/present/in-period/handed_over/left_ms}. Boxes are re-numbered: the chosen one is track 1, the rest by first appearance, noise stays NULL.",
        next: "Events run on the chosen person; every person gets its own presence and board analytics (§5); the report lands in data_quality (§10).",
        config: [
          "HANDOVER_MIN_MS 10 000 · HANDOVER_HIGH_MS 60 000",
          "LEAD_HIGH 2.0 · LEAD_MEDIUM 1.3",
          "LEFT_BRIDGE_MS 300 000 · AT_BELL_TOL_MS 10 000",
        ],
      },
    ],
  },
  {
    id: "events",
    n: 5,
    title: "Event detection",
    intro:
      "Rules over the chosen teacher's boxes and the zones. Every event has a timestamp on the video clock, so the dashboard can seek to it.",
    steps: [
      {
        name: "Presence",
        owner: "ML service · app/heuristics.py presence_intervals + classify_presence",
        in: "Her boxes' timestamps; the door zones; the duration.",
        processing:
          "Timestamps are unioned into intervals split at gaps ≥ 5 s. Each gap is then classified (§8): a door crossing seen by her movement relative to the door, an absence beyond the out-of-frame buffer, or an occlusion that is bridged back into presence.",
        out: "presence_intervals and enter/exit events, each with how it was decided.",
        next: "Presence time, entries/exits, the timeline strip, the register.",
        config: ["PRESENCE_GAP_MS 5 000", "OUT_OF_FRAME_BUFFER_MS 20 000", "END_MARGIN_MS 5 000"],
      },
      {
        name: "Board time",
        owner: "ML service · board_condition + intervals_from_samples",
        in: "Her boxes and the board zone.",
        processing:
          "A box is at the board when it intersects the board zone grown by 12% and its centre-x lies within the board's own width. A hysteresis machine turns per-sample booleans into intervals: 2 s continuously true opens one, 3 s false closes it, a single flickering sample within a 600 ms budget is tolerated, and a ≥ 5 s sampling gap hard-closes it so board time never bridges an absence.",
        out: "board_intervals, teacher_board_ms, board_enter/board_leave events.",
        next: "Time at board, board sessions, the lesson's end (§7).",
        config: [
          "BOARD_EXPAND 0.12 · BOARD_ON_MS 2 000 · BOARD_OFF_MS 3 000",
          "BOARD_FLICKER_SAMPLES 2 · BOARD_FLICKER_BUDGET_MS 600",
        ],
      },
      {
        name: "Pointing and writing",
        owner: "ML service · action_samples + action_intervals",
        in: "Her boxes; pointing and writing boxes ≥ 0.5 from the same frames.",
        processing:
          "An action box is hers when its centre lies inside her box grown by 5% in that frame; an action box with no teacher box in the frame is discarded. One sample per teacher detection, so action time is a subset of presence by construction. The same hysteresis applies with faster knobs: 600 ms on, 1 s off.",
        out: "pointing_intervals, writing_intervals, teacher_pointing_ms, teacher_writing_ms, *_start/*_end events.",
        next: "The board tiles and the lesson-start signal (§7).",
        config: [
          "ACTION_EXPAND 0.05 · ACTION_ON_MS 600 · ACTION_OFF_MS 1 000",
          "null, not 0, when the action classes were not available",
        ],
      },
      {
        name: "Where she was",
        owner: "ML service · teacher_heatmap + _movement + _track_overlay",
        in: "Her boxes.",
        processing:
          "Box centres are binned on a 32×18 grid (dwell counts). Movement is the spatial range of her centres, not path length, so jitter does not fake a walk. For playback the centre path is simplified with Ramer–Douglas–Peucker and a bbox keyframe is kept every 2 s — about 2% of the raw rows, and permanent.",
        out: "heatmap, movement, overlay {polyline, keyframes}.",
        next: "Heatmap, circulation (room covered, spread, anchor spot, a style label), the player overlay after hot rows age out.",
        config: [
          "HEATMAP_GRID_W 32 · HEATMAP_GRID_H 18",
          "OVERLAY_RDP_EPSILON 0.005 · OVERLAY_KEYFRAME_MS 2 000",
        ],
      },
      {
        name: "Every tracked adult, not just her",
        owner: "ML service · app/jobs.py _person_analytics",
        in: "Each person's boxes.",
        processing:
          "Presence (same gap and door rules) and board intervals are computed for every substantial person and stored on their track's meta. This is what lets the previous period's teacher, tracked as an 'adult' in this file, be credited to her own period (§9).",
        out: "tracks[].meta.presence_intervals / present_ms / board_intervals / board_ms.",
        next: "The register.",
      },
    ],
  },
  {
    id: "audio",
    n: 6,
    title: "Audio and voice transcription",
    intro:
      "The audio half runs in parallel on the API host and needs no GPU. It shares the video's clock because extraction starts at t = 0 without seeking.",
    steps: [
      {
        name: "Extract",
        owner: "API worker · src/jobs/analyze-audio.ts + lib/audio.ts",
        in: "The stored file (local copy first, else the presigned URL).",
        processing:
          "ffprobe checks there is an audio track at ≥ 16 kHz (else the lesson is 'skipped', a real outcome, not a failure). ffmpeg -vn -ac 1 -ar 16000 -c:a flac writes audio.flac beside the video. No -ss: audio 0 ms is video 0 ms, which is the whole basis for lining words up with boxes.",
        out: "DATA_DIR/videos/<id>/audio.flac (94 MB for 45 min).",
        next: "Transcription.",
        config: ["MIN_TRANSCRIBE_SAMPLE_RATE 16 000", "audio queue concurrency 4"],
      },
      {
        name: "Transcribe with speakers",
        owner: "API worker · lib/transcribe.ts → AssemblyAI",
        in: "The FLAC.",
        processing:
          "Uploaded and submitted with the settings that matter: speech_models ['universal-3-5-pro'], language_codes ['en','hi'] (code-switching; language_detection would pick one language for the file), speaker_labels with speaker_options 2–6 expected speakers (speaker_labels alone returned a 4.5-minute lesson as one voice), punctuate and format_text, and up to 100 classroom key terms. The transcript id is stored so a retry never pays twice. Polled every 10 s for up to an hour.",
        out: "Words with start/end ms, confidence and speaker; diarizer turns (52 for 45 min).",
        next: "Sentences.",
        config: [
          "speech_models: universal-3-5-pro",
          "language_codes: en, hi",
          "speaker_options: 2–6",
          "POLL 10 s · MAX 1 h",
        ],
      },
      {
        name: "Sentences, not turns",
        owner: "API worker · lib/segment.ts segmentSentences",
        in: "The words.",
        processing:
          "A diarizer turn runs until somebody else speaks — one was 6.5 minutes. Sentences are cut from the words at sentence-final punctuation (. ? ! । ॥), at pauses ≥ 1 s, at a speaker change, and at a cap of 60 words or 25 s. Each sentence's confidence is the mean of its words.",
        out: "426 sentences for the 45-minute lesson.",
        next: "Language, loudness, translation.",
        config: ["PAUSE_MS 1 000 · MAX_WORDS 60 · MAX_MS 25 000"],
      },
      {
        name: "Language by grammar, not script",
        owner: "API worker · lib/segment.ts languageMix / languageOf",
        in: "Each sentence's text.",
        processing:
          "The transcriber writes much of the teacher's ENGLISH in Devanagari, so script is not language. Each Devanagari word is scored against a lexicon of Hindi function and common words (है, की, को, में, और, नहीं …) and a lexicon of common English transliterations (आई, विल, वर्कशीट …); unknown Devanagari words count half toward English; Latin words are English. Ties go to Hindi.",
        out: "language 'hi' or 'en' per sentence; a word mix for splitting a code-switched sentence's time.",
        next: "R21 and the '(Teacher used Hindi)' notes.",
        config: [
          "On the real lesson: 346 English sentences to 3 Hindi for the teacher (the first, script-based reading said 72% Hindi)",
        ],
      },
      {
        name: "Loudness",
        owner: "API worker · lib/loudness.ts",
        in: "The FLAC and the sentences.",
        processing:
          "ffmpeg astats reports RMS level per 0.5-second window (5 400 windows for 45 minutes, in about a second). Each sentence takes the power mean and the peak of its voiced windows; windows below −60 dBFS are silence and do not weigh.",
        out: "rms_db and peak_db per sentence, stored.",
        next: "R17 at read time (§10).",
        config: ["WINDOW_SEC 0.5 · SILENCE_DB −60"],
      },
      {
        name: "Translate what is not English",
        owner: "API worker · lib/translate.ts → AssemblyAI LLM gateway",
        in: "Sentences carrying Devanagari.",
        processing:
          "Sent in numbered batches of 40 to an OpenAI-shaped chat endpoint on the transcription key (the account is entitled to qwen3.5-4b-32k-fast there). The reply is parsed as JSON lines or one JSON array; a bad line loses one sentence. The gateway rate-limits per minute, so batches retry with backoff, a batch that still fails is skipped, and the next run fills it from the cache of translations already stored by exact text.",
        out: "text_en per sentence.",
        next: "Every sentence shown on the page in English.",
        config: [
          "GATEWAY_MODEL qwen3.5-4b-32k-fast · BATCH 40",
          "RETRY_DELAYS 3 / 8 / 20 s",
          "the model's own 'is this Hindi' answer is not used",
        ],
      },
      {
        name: "Store",
        owner: "API worker · replaceUtterances",
        in: "The sentences with their fields.",
        processing:
          "Delete-then-insert in one transaction, in chunks of 500. is_teacher is left NULL on purpose: whose voice is hers is decided at read time from the video (§10).",
        out: "utterances rows: idx, speaker, start_ms, end_ms, text, text_en, confidence, language, rms_db, peak_db (+ label columns for the future labelling pass).",
        next: "Read-time voice report and lesson arc.",
        config: [
          "videos.audio_status: queued → extracting → transcribing → done | skipped | empty | failed",
        ],
      },
    ],
  },
  {
    id: "lesson-start",
    n: 7,
    title: "Lesson start detection",
    intro: "Two independent signals; either starts the lesson; both together raise it to observed.",
    steps: [
      {
        name: "Board signal: writing or pointing only",
        owner: "API · lib/lesson-arc.ts",
        in: "pointing_intervals ∪ writing_intervals from the video analytics.",
        processing:
          "The first interval that starts after her arrival (tolerance 2 min). Standing at the board is not a board interaction for this purpose, so board presence is not used here.",
        out: "actionMs, or null when there was none.",
        next: "Combined with the voice signal.",
      },
      {
        name: "Voice signal: meaningful teacher speech",
        owner: "API · lib/phrases.ts + lib/lesson-arc.ts",
        in: "The teacher's sentences (her voice as resolved in §10).",
        processing:
          "The first sentence after her arrival that sets a task — 'take out', 'open page', 'worksheet 41', 'let us start', 'write down', and their Devanagari spellings (टेक आउट, वर्कशीट, स्टार्ट, लिखो …). Phrase tables stand in until an LLM labels utterances; that is why the result is provisional on this signal alone.",
        out: "voiceMs with the sentence as evidence, or null.",
        next: "Combined.",
      },
      {
        name: "Decision",
        owner: "API · lib/lesson-arc.ts",
        in: "voiceMs, actionMs, the bell.",
        processing:
          "Lesson start = the earlier of the two. Both present and within 2 minutes of each other → OBSERVED. One alone → PROVISIONAL, saying which. Neither → NOT OBSERVED. Start delay (R8) = start − bell.",
        out: "R7 with state, reason, evidence; R8.",
        next: "The 'At a glance' strip and 'The lesson' card.",
        config: [
          "CORROBORATE_MS 120 000",
          "Real lesson: 09:54:49 from her words alone (4.8 min after the bell); the detector saw < 1 s of pointing that day",
        ],
      },
    ],
  },
  {
    id: "in-out",
    n: 8,
    title: "Teacher IN/OUT logic",
    intro: "The door decides direction when there is one; otherwise, time does.",
    steps: [
      {
        name: "With a door zone",
        owner: "ML service · app/heuristics.py classify_presence + door_direction",
        in: "Each gap in her presence; her boxes in the 2 s either side; the door polygon.",
        processing:
          "OUT is seen when, as she vanishes, she is moving TOWARD the door — her foot point's distance to the door's centre falls by ≥ 3% of the frame across the window — or she is standing in the doorway. IN is seen when, as she reappears, she is moving AWAY from it or is first seen in the doorway. Vanishing near the door is not enough by itself: on real footage that is as often the door frame occluding her.",
        out: "exit + enter events with method 'door', whatever the gap's length.",
        next: "Entries/exits, presence, the events table ('seen at the door').",
        config: ["DOOR_WINDOW_MS 2 000 · DOOR_EXPAND 0.15 · DOOR_APPROACH_MIN 0.03"],
      },
      {
        name: "Without a door, or when no crossing was seen",
        owner: "ML service · classify_presence",
        in: "The gap's length.",
        processing:
          "A gap of at least the out-of-frame buffer is an absence: OUT at its start, IN at its end, method 'buffer'. Anything shorter is an occlusion or a blind spot and is bridged into presence. The first sighting is always IN. The last interval ends with OUT only if a crossing was seen or at least the buffer's worth of video remained — vanishing seconds before the end is not leaving.",
        out: "Events with method 'buffer' or 'start'; bridged presence.",
        next: "Same consumers. The register and the previous period's teacher use the same rule.",
        config: [
          "OUT_OF_FRAME_BUFFER_MS 20 000 (configurable constant)",
          "END_MARGIN_MS 5 000",
          "Real lesson: three 5–16 s door-adjacent dropouts became occlusions; one 44 s absence stayed, inferred",
        ],
      },
    ],
  },
  {
    id: "periods",
    n: 9,
    title: "Period and teacher association",
    intro:
      "A lesson is a period on the classroom's day; a file is coverage of it. The timetable places files, attribution names the period's teacher, and the register credits each adult to her own period.",
    steps: [
      {
        name: "The timetable",
        owner: "API · timetable_periods, classrooms.setTimetable, Settings → Timetable",
        in: "Typed once per classroom: label, start, end, subject, teacher, class, per ISO weekday.",
        processing:
          "Breaks are the gaps between rows, so 'does a period follow this one' and 'when did the previous period end' are derived.",
        out: "The classroom's week.",
        next: "Schedule resolution.",
        config: [
          "Seeded for the handover classroom: C.T + periods 1–8, Mon–Fri (P3 09:50–10:35, P2 ends 09:25)",
        ],
      },
      {
        name: "Placing a file in its period",
        owner: "API · lib/timetable.ts resolveSchedule",
        in: "The video's recording start (or typed date), its period label and bells if typed, the day's rows.",
        processing:
          "The video's own bells win when present (an override, or a lesson from before the table). Else the day's row named by the video's period label ('Period 3', 'P3', '3' all match). Else the row the recording's wall-clock window overlaps most — no typing at all once the start time is known. Bell offsets into the recording = bell instant − recording start.",
        out: "lesson.schedule with its source; period_start_ms / period_end_ms for the ML service; the previous period's end.",
        next: "Attribution (the period window), punctuality, the previous teacher.",
      },
      {
        name: "The previous period's teacher, kept and credited",
        owner: "ML attribution + API dto.ts toPreviousTeacher + lib/register.ts",
        in: "Attribution candidates, per-person analytics, the previous period from the timetable.",
        processing:
          "Her boxes are tracked and stored like anyone's (track_no 2+, role 'adult'). She is the handed-over adult present at this period's bell. Her departure is the end of her presence run at the bell (not her last sighting, which can be a later re-link to someone in similar clothes). Her over-run = departure − her own bell, with the break before this period named. Her presence and board minutes in this file are read from her track's meta. None of it enters this period's KPI: every period-3 number is computed on the chosen teacher's boxes only, and her presence in the KPI equals track 1's presence exactly.",
        out: "previousTeacher on the lesson DTO; a spill-over block on the previous period's register row.",
        next: "The Attendance card and the Register tab.",
        config: [
          "Real day: period 2 = no file, departure 09:55 from the period-3 file, over-ran her 09:25 bell by 30.6 min (25 the break), 6.9 min present and 4.5 at the board there",
        ],
      },
      {
        name: "The register",
        owner: "API · lessons.day → lib/register.ts buildDay",
        in: "The day's timetable rows and the day's files with their DTOs.",
        processing:
          "One row per period. Own file(s): the recordings the timetable places in it, whose attributed teacher's numbers are the period's numbers. Spill-over: the next period's file where an adult at that bell left while the next teacher remained. Arrival and departure each carry observed / not observed with the reason ('No recording covers this period').",
        out: "Rows with arrival, departure (own or from the next recording), over-/under-run, presence.",
        next: "The Register tab.",
      },
    ],
  },
  {
    id: "kpis",
    n: 10,
    title: "Analytics and KPI generation",
    intro:
      "Raw boxes → intervals and events (ML, stored) → numbers (API, at read time). Nothing in the second step is stored, so a changed definition is a re-read, never a re-run. The table lists every measurement with its source data and its calculation.",
    steps: [],
  },
  {
    id: "backend",
    n: 11,
    title: "Backend, storage and communication",
    intro:
      "Four services and two stores. All communication is HTTP with polling; there is no message broker between services — the queue is BullMQ on Redis inside the API.",
    steps: [
      {
        name: "API service",
        owner: "Bun · Hono · oRPC · Drizzle · BullMQ — apps/api-service",
        in: "Browser requests; ML results; provider callbacks by polling.",
        processing:
          "HTTP: POST /videos (upload), GET /videos/:id/stream (range-served from the local cache, pulled from S3 on demand), /thumbnail, /health, /auth/*. RPC under /rpc/*: videos (list, get, setLessonDetails, status, detections, delete), classrooms (list, create, get, update, delete, setZones, setTimetable, metrics), analysis (reanalyze, reanalyzeAudio, rederive), lessons (day), zones (upsert), board (detect), settings (get, update), gpu (status, catalog, image, pods, create, terminate, start, stop, createVolume). The same process runs the BullMQ workers: video-analysis concurrency 1, audio-analysis concurrency 4. Auth is a signed cookie whose secret is the admin password.",
        out: "Typed responses; the frontend imports the router's types, so a DTO change is a compile error on the page.",
        next: "UI (§12).",
        config: [
          "oRPC contract shared as @classroom/api-contracts",
          "bun --hot does not re-register workers: restart after editing job code",
        ],
      },
      {
        name: "Postgres + TimescaleDB",
        owner: "classroom-db container (Postgres 16, TimescaleDB 2.28)",
        in: "Rows from the API and COPY from the ML service.",
        processing:
          "Tables: classrooms, classroom_zones, timetable_periods, videos (status, lesson fields, audio fields), zones, tracks (per person: role, first/last, meta with overlay and analytics), events, video_analytics (presence/board/action intervals, heatmap, data_quality jsonb), utterances, app_settings, detection_events. detection_events is a hypertable on wall-clock ts, compressed after 24 h (segment by video_id, ordered by video_ts_ms, track_no) and DROPPED after 2 days: it exists for cheap re-derives, while tracks.meta keeps the permanent overlay.",
        out: "Everything the dashboard reads.",
        next: "The API's read-time derivations.",
        config: [
          "18 hand-written migrations (drizzle/), applied by bun run db:migrate",
          "hot tier retention 2 days · compression 24 h",
        ],
      },
      {
        name: "Object store",
        owner: "MinIO (S3 API) on-site — classroom-minio",
        in: "Uploaded videos and thumbnails.",
        processing:
          "Bun's S3 client writes as the upload streams; readers get presigned GET URLs (6 h for analysis). The pod reaches it through a reverse SSH tunnel on port 9000 under the same host name, because SigV4 signs the Host header. Student video never leaves the site except through that tunnel to the rented GPU for the length of one analysis.",
        out: "Blobs by key videos/<id>/original.mp4 and thumb.jpg.",
        next: "Probe, stream, ML fetch.",
      },
      {
        name: "ML service",
        owner:
          "FastAPI · PyTorch · RF-DETR · ffmpeg/cv2 — services/ml-service, image ghcr.io/…/ml-service:feat-rfdetr-pipeline",
        in: "POST /analyze (async, idempotent), POST /rederive (sync replay of stored rows), POST /detect-board and /detect-door (zone editor), GET /health, GET /jobs/{id}, GET /jobs/{id}/result.",
        processing:
          "Runs on a RunPod GPU the app provisions itself from Settings (RunPod REST + GraphQL catalog; CUDA 13 pin; the pod's proxy hostname becomes the ML URL). Writes detections straight to Postgres over the 5533 tunnel; returns everything else as one JSON validated by pydantic models — an unknown key is dropped silently, so every new field is added to the model in the same commit.",
        out: "AnalysisResult JSON; detection_events rows.",
        next: "API ingest: replaceDerived writes tracks, events and video_analytics in one transaction, carrying forward action KPIs a teacher-only re-derive cannot recompute.",
        config: [
          "Locally: uvicorn on :8000 on Apple MPS for re-derives; no hot reload",
          "Pod: RTX A4000 $0.25/hr worked; 45 min of 1080p ≈ 17 min detection ≈ $0.14",
        ],
      },
      {
        name: "Redis",
        owner: "classroom-redis",
        in: "Job payloads {videoId, attemptId}.",
        processing:
          "Two BullMQ queues, retries with exponential backoff, stalled-job recovery; job progress mirrored into videos.progress for the UI.",
        out: "—",
        next: "—",
      },
    ],
  },
  {
    id: "ui",
    n: 12,
    title: "Final output and UI",
    intro:
      "The dashboard reads one typed detail object per lesson and one per classroom-day. Every number links to the moment in the video it came from.",
    steps: [
      {
        name: "The lesson detail",
        owner: "API · videos.get → router/dto.ts toDetailDto",
        in: "videos row, zones, tracks, events, video_analytics, the classroom's timetable, utterances.",
        processing:
          "Built at read time, in this order: resolve the schedule (§9) → voice report (teacher's voice, talk split, pace, questions, languages, raised voice, coverage) → lesson arc (R7–R19 from her sentences, the board actions, the bells, her arrival and departure) → punctuality (R1–R5 from presence intervals against the bells; withheld when two adults blend and attribution is not high) → previous teacher → trust (all 23 with states) → transcript rows with isTeacher and English.",
        out: "One JSON the page renders: video, lesson + schedule, punctuality, previousTeacher, voice, arc, trust, transcript, zones, tracks (with overlays), events, analytics.",
        next: "The lesson page.",
        config: [
          "React Query polls videos.status every 2 s until done; videos.get refetches on every mutation",
        ],
      },
      {
        name: "The lesson page",
        owner: "Frontend · routes/videos.$id.tsx and its cards",
        in: "The detail object; the video stream; detections for the overlay.",
        processing:
          "Top to bottom: the video with Detections/Raw overlay (boxes per track from videos.detections, sampled to the player's fps; students never drawn) → At a glance (arrived, lesson started, lesson ended, left, in the room) → Attendance (the editable lesson details, absences with how each was decided, the previous period's teacher) → The lesson (R7–R19 as plain-language stats, sentences behind a toggle) → Voice → At the board and Board sessions → Movement (timeline, circulation, heatmap) → Evidence (events with 'seen at the door' / 'inferred') → Transcript in English with Hindi notes → Reliability (quality tiers, notes, and the full list of 23). Every time on the page seeks the player.",
        out: "The screen.",
        next: "—",
        config: [
          "One stat grammar everywhere: label, value, one line of context, state (Observed / Provisional / Not observed), reason on hover, requirement id as tooltip",
        ],
      },
      {
        name: "The classroom",
        owner: "Frontend · routes/classrooms.$id.*",
        in: "classrooms.get, videos.list, lessons.day, classrooms.metrics.",
        processing:
          "Lessons (upload + list with status), Register (the day as periods, §9), Metrics (aggregates over analysed lessons), Configuration (details, zone template drawn on a real frame, the timetable editor, delete).",
        out: "—",
        next: "—",
      },
      {
        name: "Settings",
        owner: "Frontend · routes/settings.tsx → settings/gpu routers",
        in: "Keys and the pod spec.",
        processing:
          "AssemblyAI and RunPod keys, the school timezone, the pod spec resolved from the live catalog (GPU type, region, image tag, CUDA pin, vCPU per GPU), the pod's SSH key, Create/Terminate, and a preflight that inspects the configured image tag.",
        out: "app_settings rows.",
        next: "—",
      },
    ],
  },
];

export interface KpiRow {
  id: string;
  name: string;
  source: string;
  calc: string;
  state: string;
}

export const KPIS: KpiRow[] = [
  {
    id: "R1",
    name: "Arrival time",
    source: "presence_intervals[0].start + recording start",
    calc: "Wall clock of the first presence interval of the attributed teacher.",
    state: "Observed when anchored and not blended",
  },
  {
    id: "R2",
    name: "Arrival against the bell",
    source: "R1, timetable bell",
    calc: "R1 − scheduled start, minutes to one decimal; positive is late.",
    state: "Needs the bells",
  },
  {
    id: "R3",
    name: "Departure time",
    source: "last presence interval end",
    calc: "Wall clock of the end of her last presence interval.",
    state: "Observed",
  },
  {
    id: "R4",
    name: "Departure against the bell",
    source: "R3, timetable bell",
    calc: "R3 − scheduled end; positive is late.",
    state: "Needs the bells",
  },
  {
    id: "R5",
    name: "Time in the room",
    source: "teacher_present_ms, period length",
    calc: "Sum of presence intervals ÷ (end bell − start bell).",
    state: "Needs the bells",
  },
  {
    id: "R6",
    name: "Mid-lesson absences",
    source: "entry_exit",
    calc: "Count of exits, each labelled seen-at-the-door or inferred (§8).",
    state: "Observed",
  },
  {
    id: "R7",
    name: "Lesson start",
    source: "teacher sentences, pointing/writing intervals",
    calc: "Earlier of the first task-setting sentence and the first board action; observed when both agree within 2 min.",
    state: "Provisional on one signal",
  },
  {
    id: "R8",
    name: "Start delay",
    source: "R7, bell",
    calc: "R7 − scheduled start.",
    state: "As R7",
  },
  {
    id: "R9",
    name: "Lesson end",
    source: "teacher sentences, board_intervals, departure",
    calc: "Later of the last teaching sentence (by exclusion of procedure, attention, pack-up, homework, continuation) and the last board interval end, capped at departure.",
    state: "Provisional",
  },
  { id: "R10", name: "Lesson duration", source: "R7, R9", calc: "R9 − R7.", state: "Provisional" },
  {
    id: "R11",
    name: "Fit the period",
    source: "R7, R9, bells",
    calc: "R7 ≥ start bell − 1 min and R9 ≤ end bell + 1 min.",
    state: "Provisional",
  },
  {
    id: "R12",
    name: "Over- or under-run",
    source: "R9, end bell",
    calc: "R9 − scheduled end; positive past the bell, negative unused.",
    state: "Provisional",
  },
  {
    id: "R13",
    name: "Closure and its type",
    source: "teacher sentences in the last 5 min before R9",
    calc: "First match of review / summary / reflection / exit-question phrases; 'none' is a reported outcome.",
    state: "Provisional",
  },
  {
    id: "R14",
    name: "Continuation",
    source: "teacher sentences",
    calc: "Any sentence saying the topic continues next time.",
    state: "Provisional",
  },
  {
    id: "R15",
    name: "Homework set",
    source: "teacher sentences",
    calc: "Any homework phrase, in either script; time of the first.",
    state: "Provisional",
  },
  {
    id: "R16",
    name: "Pack-up instruction",
    source: "—",
    calc: "Removed from the page at the user's request; the phrase label survives only to exclude such sentences from R9.",
    state: "Not shown",
  },
  {
    id: "R17",
    name: "Raised-voice events",
    source: "utterances.rms_db of her sentences",
    calc: "Sentences ≥ 6 dB over her own median for ≥ 1.5 s, merged within 5 s; count and per 10 min.",
    state: "Observed",
  },
  {
    id: "R18",
    name: "Attention requests",
    source: "teacher sentences",
    calc: "Sentences matching attention cues ('listen', 'quiet', सुनो …); count and per 10 min of the lesson.",
    state: "Provisional",
  },
  {
    id: "R19",
    name: "Off-lesson drift",
    source: "teacher sentences",
    calc: "Runs of ≥ 2 procedural sentences within 30 s (notebooks, planners, signatures …) stand in: episodes and total time.",
    state: "Provisional",
  },
  {
    id: "R20",
    name: "Questions asked",
    source: "teacher sentences",
    calc: "Sentences ending in '?' minus check-ins (whole-sentence or tagged: 'ठीक है?', '…, okay?'); per 10 min; the list is shown.",
    state: "Provisional",
  },
  {
    id: "R21",
    name: "Languages used",
    source: "languageMix per sentence",
    calc: "Her speech time split by each sentence's Hindi/English word mix; a language counts as used at ≥ 5%; switches between consecutive sentences per minute.",
    state: "Observed",
  },
  {
    id: "R22",
    name: "Observation coverage",
    source: "data_quality; transcribed span",
    calc: "Video: share of sampled frames she was found in, tiers ≥ 0.85 / ≥ 0.6; continuity breaks ≤ 6 / ≤ 20; detection confidence ≥ 0.7 / ≥ 0.5; attribution tier. Audio: transcribed share and mean confidence.",
    state: "Observed",
  },
  {
    id: "R23",
    name: "Not observed",
    source: "every measure's state",
    calc: "Count of measurements withheld rather than guessed; each with its reason.",
    state: "Observed",
  },
];

export const VOICE_EXTRAS: KpiRow[] = [
  {
    id: "—",
    name: "Teacher's voice",
    source: "utterances × presence_intervals",
    calc: "The diarized speaker carrying most speech while the attributed teacher is in the room: high at ≥ 60% share and a 2× lead over the next voice, medium at ≥ 45% and 1.3×, else nobody.",
    state: "Tiered",
  },
  {
    id: "—",
    name: "Talk split",
    source: "her sentences, all sentences, duration",
    calc: "Union of her sentence spans ÷ duration; others likewise; the rest is no speech.",
    state: "Observed",
  },
  {
    id: "—",
    name: "Longest stretch",
    source: "her sentences",
    calc: "Longest chain of her sentences with gaps < 2 s and nobody else in between.",
    state: "Observed",
  },
  {
    id: "—",
    name: "Pace",
    source: "her words, her speech time",
    calc: "Words per minute of her own speech time.",
    state: "Observed",
  },
];
