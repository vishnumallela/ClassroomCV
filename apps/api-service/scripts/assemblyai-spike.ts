/**
 * AssemblyAI spike — transcribe one lesson, off to the side of the pipeline.
 *
 * Deliberately touches nothing: no queue, no database, no ML service. It answers
 * the questions you have to answer before wiring audio into the real run —
 * how long a lesson actually takes to transcribe, whether diarization can pick
 * the teacher out of a classroom, and what the JSON gives you for
 * docs/domain-a-kpis-and-labels.md group 3.
 *
 *   ASSEMBLYAI_API_KEY=... bun apps/api-service/scripts/assemblyai-spike.ts \
 *     data/videos/<id>/source.mp4 [--out DIR] [--speakers N]
 *
 * The input may also be an https URL AssemblyAI can reach itself, in which case
 * nothing is extracted or uploaded. A local MinIO presigned URL is NOT such a
 * URL — it has to be publicly resolvable.
 *
 * Endpoints follow the AssemblyAI v2 REST API; if a call 404s, check their
 * current docs before assuming the key is wrong.
 */

const API = "https://api.assemblyai.com/v2";
const POLL_INTERVAL_MS = 5_000;

interface Utterance {
  speaker: string;
  text: string;
  start: number;
  end: number;
  confidence: number;
}

interface Transcript {
  id: string;
  status: "queued" | "processing" | "completed" | "error";
  error?: string | null;
  text?: string | null;
  audio_duration?: number | null;
  utterances?: Utterance[] | null;
  words?: { text: string; start: number; end: number; speaker: string | null }[] | null;
}

function die(message: string): never {
  console.error(`error: ${message}`);
  process.exit(1);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function fmtMs(ms: number): string {
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}m${String(s).padStart(2, "0")}s`;
}

async function spawn(cmd: string[]): Promise<{ code: number; stderr: string }> {
  const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe" });
  const stderr = await new Response(proc.stderr).text();
  return { code: await proc.exited, stderr };
}

/**
 * 16 kHz mono FLAC: exactly what an ASR model consumes, so nothing is lost by
 * downmixing, and a 37-minute lesson goes from a ~2 GB mp4 to tens of MB — the
 * difference between a minute of upload and twenty. FLAC rather than AAC so the
 * same file can be reused for the energy analysis AssemblyAI does not do
 * (raised voice, room noise), where a lossy codec would be arguing with you.
 *
 * No -ss, no -t: the audio must keep the video's t=0 or every timestamp you get
 * back is offset against the detection timeline it has to be read alongside.
 */
async function extractAudio(input: string, outPath: string): Promise<void> {
  const { code, stderr } = await spawn([
    "ffmpeg",
    "-y",
    "-i",
    input,
    "-vn",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-c:a",
    "flac",
    outPath,
  ]);
  if (code !== 0) die(`ffmpeg exited ${code}: ${stderr.slice(-400)}`);
}

async function upload(path: string, key: string): Promise<string> {
  const file = Bun.file(path);
  const res = await fetch(`${API}/upload`, {
    method: "POST",
    headers: { authorization: key },
    body: file,
  });
  if (!res.ok) die(`upload failed: ${res.status} ${(await res.text()).slice(0, 300)}`);
  const body = (await res.json()) as { upload_url?: string };
  if (!body.upload_url) die("upload returned no upload_url");
  return body.upload_url;
}

async function submit(audioUrl: string, key: string, speakers: number | null): Promise<string> {
  const res = await fetch(`${API}/transcript`, {
    method: "POST",
    headers: { authorization: key, "content-type": "application/json" },
    body: JSON.stringify({
      audio_url: audioUrl,
      // X-7 in docs/domain-a-requirements.md: nothing in A2 can be scored
      // until the teacher's voice is separable from the room's.
      speaker_labels: true,
      ...(speakers ? { speakers_expected: speakers } : {}),
      punctuate: true,
      format_text: true,
    }),
  });
  if (!res.ok) die(`submit failed: ${res.status} ${(await res.text()).slice(0, 300)}`);
  const body = (await res.json()) as { id?: string };
  if (!body.id) die("submit returned no transcript id");
  return body.id;
}

async function poll(id: string, key: string): Promise<Transcript> {
  const startedAt = Date.now();
  for (;;) {
    const res = await fetch(`${API}/transcript/${id}`, { headers: { authorization: key } });
    if (!res.ok) die(`poll failed: ${res.status} ${(await res.text()).slice(0, 300)}`);
    const body = (await res.json()) as Transcript;
    if (body.status === "completed" || body.status === "error") return body;
    console.log(`  ${body.status}… ${fmtMs(Date.now() - startedAt)} elapsed`);
    await sleep(POLL_INTERVAL_MS);
  }
}

/** Talk time per speaker — the raw material for KPI 20 (teacher talk share). */
function speakerTable(utterances: Utterance[]): { speaker: string; ms: number; turns: number }[] {
  const totals = new Map<string, { ms: number; turns: number }>();
  for (const u of utterances) {
    const prev = totals.get(u.speaker) ?? { ms: 0, turns: 0 };
    totals.set(u.speaker, { ms: prev.ms + (u.end - u.start), turns: prev.turns + 1 });
  }
  return [...totals.entries()]
    .map(([speaker, v]) => ({ speaker, ...v }))
    .toSorted((a, b) => b.ms - a.ms);
}

function summarize(t: Transcript, wallMs: number): void {
  const utterances = t.utterances ?? [];
  const audioMs = (t.audio_duration ?? 0) * 1000;

  console.log(`\ntranscript ${t.id}`);
  console.log(`  audio          ${fmtMs(audioMs)}`);
  console.log(
    `  wall time      ${fmtMs(wallMs)}` +
      (audioMs > 0 ? `  (${(audioMs / wallMs).toFixed(1)}x realtime)` : ""),
  );
  console.log(`  words          ${t.words?.length ?? 0}`);
  console.log(`  utterances     ${utterances.length}`);

  if (utterances.length === 0) {
    console.log("\n  No utterances. Either the audio is silent or diarization was off.");
    return;
  }

  const speakers = speakerTable(utterances);
  const spokenMs = speakers.reduce((sum, s) => sum + s.ms, 0);
  console.log("\n  speaker   talk time   share of speech   turns");
  for (const s of speakers) {
    const share = spokenMs > 0 ? ((s.ms / spokenMs) * 100).toFixed(1) : "0.0";
    console.log(
      `  ${s.speaker.padEnd(9)} ${fmtMs(s.ms).padStart(8)}   ${share.padStart(9)}%   ${String(s.turns).padStart(5)}`,
    );
  }

  // Which of these is the teacher is NOT answered here on purpose. Loudest
  // talker is a guess; the pipeline already knows when the teacher was at the
  // board, and that is the honest way to anchor a label to a speaker.
  console.log("\n  first three turns:");
  for (const u of utterances.slice(0, 3)) {
    console.log(`  [${fmtMs(u.start)}] ${u.speaker}: ${u.text.slice(0, 90)}`);
  }
  const last = utterances.at(-1);
  if (last) {
    console.log(`  last turn:`);
    console.log(`  [${fmtMs(last.start)}] ${last.speaker}: ${last.text.slice(0, 90)}`);
  }

  const questions = utterances.filter((u) => u.text.includes("?")).length;
  console.log(`\n  turns containing a question: ${questions}  (KPI 25, rough cut)`);
  console.log(
    "  no loudness anywhere in this payload — raised voice (KPI 18) and off-task\n" +
      "  noise (KPI 23) are energy measurements and need their own pass over the FLAC.",
  );
}

async function main(): Promise<void> {
  const key = process.env.ASSEMBLYAI_API_KEY;
  if (!key) die("set ASSEMBLYAI_API_KEY");

  const args = process.argv.slice(2);
  const input = args.find((a) => !a.startsWith("--"));
  if (!input) die("usage: bun apps/api-service/scripts/assemblyai-spike.ts <video|audio|url>");

  const outDir = args[args.indexOf("--out") + 1] ?? ".";
  const speakersArg = args.includes("--speakers")
    ? Number(args[args.indexOf("--speakers") + 1])
    : null;
  const speakers = speakersArg && Number.isFinite(speakersArg) ? speakersArg : null;

  const startedAt = Date.now();
  let audioUrl: string;
  let audioPath: string | null = null;

  if (input.startsWith("https://")) {
    console.log(`using remote audio ${input}`);
    audioUrl = input;
  } else {
    if (!(await Bun.file(input).exists())) die(`no such file: ${input}`);
    audioPath = `${outDir}/spike-audio.flac`;
    console.log(`extracting 16 kHz mono audio → ${audioPath}`);
    const t0 = Date.now();
    await extractAudio(input, audioPath);
    const bytes = Bun.file(audioPath).size;
    console.log(`  ${(bytes / 1e6).toFixed(1)} MB in ${fmtMs(Date.now() - t0)}`);

    console.log("uploading");
    const t1 = Date.now();
    audioUrl = await upload(audioPath, key);
    console.log(`  uploaded in ${fmtMs(Date.now() - t1)}`);
  }

  const id = await submit(audioUrl, key, speakers);
  console.log(`submitted ${id}; polling every ${POLL_INTERVAL_MS / 1000}s`);
  const transcript = await poll(id, key);

  if (transcript.status === "error") die(`transcription failed: ${transcript.error ?? "unknown"}`);

  const jsonPath = `${outDir}/spike-transcript-${id}.json`;
  await Bun.write(jsonPath, JSON.stringify(transcript, null, 2));
  summarize(transcript, Date.now() - startedAt);
  console.log(`\nfull payload: ${jsonPath}`);
  // The FLAC is kept on purpose: the energy pass reads the same file, and
  // re-extracting it from a 2 GB mp4 to try one threshold is a waste.
  if (audioPath) console.log(`audio kept:   ${audioPath}`);
}

main().catch((err: unknown) => die(err instanceof Error ? err.message : String(err)));
