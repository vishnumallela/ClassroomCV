import { getAppSettings } from "@api/lib/app-settings";

/**
 * Transcription and diarization (AssemblyAI), the input to Groups C and D of
 * docs/teacher-measurements.md.
 *
 * Submit-then-poll, exactly like the GPU path: the work happens on someone
 * else's machine and this process only waits. That is what makes running the
 * two halves concurrently nearly free — see docs/teacher-measurements.md and
 * the audio-analysis queue.
 */

const API = "https://api.assemblyai.com/v2";

/**
 * Below this, don't spend the money.
 *
 * Sampling at 8 kHz caps the signal at 4 kHz and removes the band that
 * distinguishes /s/, /f/ and /sh/ — and Hindi's aspirated and retroflex
 * consonants with them. 31 of the 39 recordings in the archive are 8 kHz, and
 * transcribing them would produce numbers that look like measurements while
 * really reporting the microphone. New recordings at 16 kHz+ are the target;
 * older ones are skipped with a reason rather than guessed at.
 */
export const MIN_TRANSCRIBE_SAMPLE_RATE = 16_000;

export class TranscribeNotConfiguredError extends Error {}

async function apiKey(): Promise<string> {
  const settings = await getAppSettings().catch(() => ({}) as Record<string, never>);
  const key = settings.assemblyaiApiKey?.trim() || process.env.ASSEMBLYAI_API_KEY?.trim();
  if (!key) {
    throw new TranscribeNotConfiguredError(
      "No AssemblyAI API key. Add one on the Settings page, or set ASSEMBLYAI_API_KEY.",
    );
  }
  return key;
}

export interface Utterance {
  speaker: string;
  text: string;
  start: number;
  end: number;
  confidence: number;
}

export interface TranscriptResult {
  status: "queued" | "processing" | "completed" | "error";
  error: string | null;
  utterances: Utterance[];
  audioDurationMs: number | null;
  /** Which languages the model actually heard, when it reports them. */
  detectedLanguages: string[];
}

interface RawTranscript {
  id?: string;
  status?: string;
  error?: string | null;
  audio_duration?: number | null;
  utterances?: Utterance[] | null;
  language_code?: string | null;
  language_detection_results?: { language_code?: string }[] | null;
}

async function readError(res: Response): Promise<string> {
  try {
    return (await res.text()).slice(0, 300);
  } catch {
    return `${res.status}`;
  }
}

/**
 * Terms worth boosting, and why these specifically.
 *
 * Not transcript polish. The measurements in Group C and D are found BY these
 * phrases — an attention request (R18) or a pack-up instruction (R16) is
 * identified from what was said, so a phrase the model mishears is a
 * measurement that silently never happens. Two real misses from the first test
 * run: "ma'am" came back as मामा, and "डाँट" (a scolding) as the English word
 * "dart", which would have dropped a behaviour-talk episode entirely.
 *
 * Kept deliberately short. Boosting trades false negatives for false positives,
 * and a hallucinated cue is a phantom attention request in the KPI.
 */
const CLASSROOM_KEY_TERMS = [
  "ma'am",
  "sir",
  "बच्चे",
  "डाँट",
  "shaant ho jao",
  "chup ho jao",
  "dhyan do",
  "idhar dekho",
  "baith jao",
  "kitab kholo",
  "copy mein likho",
  "page number",
  "samajh aaya",
  "haath uthao",
  "bag pack karo",
  "bell baj gayi",
  "homework",
];

export interface SubmitOptions {
  /** Extra vocabulary for this lesson — subject terms, the teacher's name. */
  keyTerms?: string[];
  /**
   * Free-text context for the acoustic model.
   *
   * Used ONLY when no key terms are in play: the API rejects a request
   * carrying both `prompt` and `keyterms_prompt`. Key terms win by default —
   * they are the documented choice when the specific vocabulary is known, and
   * here it is: the phrases the measurements are found by.
   */
  prompt?: string;
}

/** ≤100 terms, ≤50 characters each — longer ones are dropped, more than 100 errors. */
const MAX_KEY_TERMS = 100;
const MAX_KEY_TERM_CHARS = 50;

function normalizeKeyTerms(extra: string[] = []): string[] {
  const seen = new Set<string>();
  for (const term of [...CLASSROOM_KEY_TERMS, ...extra]) {
    const trimmed = term.trim();
    if (trimmed && trimmed.length <= MAX_KEY_TERM_CHARS) seen.add(trimmed);
  }
  return [...seen].slice(0, MAX_KEY_TERMS);
}

/** Upload local audio and start a transcript. Returns the transcript id. */
export async function submitTranscript(
  audioPath: string,
  options: SubmitOptions = {},
): Promise<string> {
  const key = await apiKey();
  const keyTerms = normalizeKeyTerms(options.keyTerms);

  const uploadRes = await fetch(`${API}/upload`, {
    method: "POST",
    headers: { authorization: key },
    body: Bun.file(audioPath),
  });
  if (!uploadRes.ok) throw new Error(`audio upload failed: ${await readError(uploadRes)}`);
  const { upload_url: uploadUrl } = (await uploadRes.json()) as { upload_url?: string };
  if (!uploadUrl) throw new Error("audio upload returned no url");

  const res = await fetch(`${API}/transcript`, {
    method: "POST",
    headers: { authorization: key, "content-type": "application/json" },
    body: JSON.stringify({
      audio_url: uploadUrl,
      // Both codes, English required by the API. Classrooms here code-switch
      // mid-sentence, and pinning a single language mangles whichever half
      // loses — verified on a real lesson.
      language_codes: ["en", "hi"],
      // R21 needs to know which language each turn was in.
      language_detection: true,
      // X-7: nothing in Group D is measurable until the teacher's voice is
      // separable from the room's.
      speaker_labels: true,
      // Restarts and hesitation are not noise here — "repeated attempts to
      // begin" is a graded behaviour, and cleaning them up makes a floundering
      // start read as a fluent one.
      disfluencies: true,
      punctuate: true,
      format_text: true,
      // Exactly one of these, never both — the API rejects a request carrying
      // `prompt` alongside `keyterms_prompt`.
      ...(keyTerms.length > 0
        ? { keyterms_prompt: keyTerms }
        : options.prompt
          ? { prompt: options.prompt }
          : {}),
    }),
  });
  if (!res.ok) throw new Error(`transcript submit failed: ${await readError(res)}`);
  const body = (await res.json()) as RawTranscript;
  if (!body.id) throw new Error("transcript submit returned no id");
  return body.id;
}

export async function getTranscript(id: string): Promise<TranscriptResult> {
  const key = await apiKey();
  const res = await fetch(`${API}/transcript/${id}`, { headers: { authorization: key } });
  if (!res.ok) throw new Error(`transcript fetch failed: ${await readError(res)}`);
  const body = (await res.json()) as RawTranscript;

  const status = body.status ?? "error";
  const languages = new Set<string>();
  if (body.language_code) languages.add(body.language_code);
  for (const entry of body.language_detection_results ?? []) {
    if (entry.language_code) languages.add(entry.language_code);
  }

  return {
    status:
      status === "completed" || status === "queued" || status === "processing" ? status : "error",
    error: body.error ?? null,
    utterances: body.utterances ?? [],
    audioDurationMs: body.audio_duration ? Math.round(body.audio_duration * 1000) : null,
    detectedLanguages: [...languages],
  };
}
