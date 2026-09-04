import { getAppSettings } from "@api/lib/app-settings";

/**
 * English for the sentences that are not already English.
 *
 * The transcriber writes much of the teacher's ENGLISH in Devanagari script
 * ("आई विल साइन एंड देन रिटर्न" is "I will sign and then return"), and the
 * rest is Hindi. Both must read as English on the page. A language model does
 * this through AssemblyAI's LLM gateway, which runs on the transcription key
 * already configured, so no second key is needed.
 *
 * Only sentences carrying Devanagari are sent, in numbered batches, and the
 * reply is one JSON object per line so a single malformed line loses one
 * sentence rather than the batch. The model is also asked whether the
 * sentence was actually Hindi — a second opinion beside lib/segment.ts, and
 * the one the "(Teacher used Hindi)" note rests on when both agree.
 */

export class TranslateNotConfiguredError extends Error {}

export const GATEWAY = "https://llm-gateway.assemblyai.com/v1/chat/completions";
/**
 * The gateway model this account can use (its /v1/models lists Claude and
 * Gemini too, but the account is not entitled to them). A 4B model is enough
 * for sentence-level translation and de-transliteration; its own "is this
 * Hindi" answer is not trusted — lib/segment.ts decides that.
 */
export const GATEWAY_MODEL = "qwen3.5-4b-32k-fast";
export const BATCH = 40;

const DEVANAGARI = /[ऀ-ॿ]/u;

export function needsTranslation(text: string): boolean {
  return DEVANAGARI.test(text);
}

export interface Translated {
  en: string;
  /** The model's own view of whether the sentence was Hindi (not transliterated English). */
  hindi: boolean | null;
}

const PROMPT =
  "You are translating a classroom transcript from an Indian school. Each numbered line is one " +
  "sentence a teacher or student said. Some lines are Hindi; many are ENGLISH that the " +
  "transcriber wrote in Devanagari script (for example 'आई विल' is 'I will', 'वर्कशीट' is " +
  "'worksheet'). For EVERY line output exactly one JSON object on its own line, in order: " +
  '{"i": <line number>, "en": "<natural English, keeping names as names>", "hindi": <true only ' +
  "if the sentence is actually Hindi, false if it is English written in Devanagari or mixed>}. " +
  "Output nothing else — no prose, no code fences.";

/** Parse the model's reply — JSON lines, or one JSON array — tolerant of
 *  stray prose and fences. */
export function parseTranslations(reply: string): Map<number, Translated> {
  const out = new Map<number, Translated>();
  const stripped = reply.replace(/```(json)?/g, "").trim();
  if (stripped.startsWith("[")) {
    try {
      const arr = JSON.parse(stripped) as unknown;
      if (Array.isArray(arr)) {
        for (const obj of arr as { i?: unknown; en?: unknown; hindi?: unknown }[]) {
          if (typeof obj.i === "number" && typeof obj.en === "string" && obj.en.trim()) {
            out.set(obj.i, {
              en: obj.en.trim(),
              hindi: typeof obj.hindi === "boolean" ? obj.hindi : null,
            });
          }
        }
        return out;
      }
    } catch {
      // fall through to the line-by-line read
    }
  }
  for (const raw of reply.split("\n")) {
    const line = raw
      .trim()
      .replace(/^```(json)?|```$/g, "")
      .trim();
    if (!line.startsWith("{")) continue;
    try {
      const obj = JSON.parse(line) as { i?: unknown; en?: unknown; hindi?: unknown };
      if (typeof obj.i === "number" && typeof obj.en === "string" && obj.en.trim()) {
        out.set(obj.i, {
          en: obj.en.trim(),
          hindi: typeof obj.hindi === "boolean" ? obj.hindi : null,
        });
      }
    } catch {
      // one bad line loses one sentence, not the batch
    }
  }
  return out;
}

async function apiKey(): Promise<string> {
  const settings = await getAppSettings().catch(() => ({}) as Record<string, string | undefined>);
  const key = settings.assemblyaiApiKey?.trim();
  if (!key) throw new TranslateNotConfiguredError("No AssemblyAI key is configured in Settings.");
  return key;
}

/** Back off on 429s and transient failures; the gateway rate-limits per minute. */
export const RETRY_DELAYS_MS = [3_000, 8_000, 20_000];

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function complete(prompt: string, key: string): Promise<string> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
    const res = await fetch(GATEWAY, {
      method: "POST",
      headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: GATEWAY_MODEL,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 4000,
        temperature: 0,
      }),
    });
    if (res.ok) {
      const body = (await res.json()) as { choices?: { message?: { content?: string } }[] };
      return body.choices?.[0]?.message?.content ?? "";
    }
    lastError = new Error(`translation failed: ${res.status} ${(await res.text()).slice(0, 300)}`);
    const retryable = res.status === 429 || res.status >= 500;
    const delay = RETRY_DELAYS_MS[attempt];
    if (!retryable || delay === undefined) break;
    await sleep(delay);
  }
  throw lastError ?? new Error("translation failed");
}

/**
 * Translate the given sentences; returns idx -> translation for those the
 * model answered. Sentences without Devanagari are never sent.
 */
export async function translateSentences(
  sentences: { idx: number; text: string }[],
): Promise<Map<number, Translated>> {
  const todo = sentences.filter((s) => needsTranslation(s.text));
  const out = new Map<number, Translated>();
  if (todo.length === 0) return out;
  const key = await apiKey();
  // A batch that fails after its retries is skipped, not fatal: whatever was
  // translated is stored, and the next run fills the gaps from the cache of
  // what already succeeded.
  let failed = 0;
  for (let i = 0; i < todo.length; i += BATCH) {
    const batch = todo.slice(i, i + BATCH);
    const numbered = batch.map((s, n) => `${n + 1}. ${s.text}`).join("\n");
    try {
      const reply = await complete(`${PROMPT}\n\n${numbered}`, key);
      const parsed = parseTranslations(reply);
      for (const [n, t] of parsed) {
        const s = batch[n - 1];
        if (s) out.set(s.idx, t);
      }
    } catch (err) {
      failed++;
      if (failed > 2) throw err;
    }
    // Spread the batches out; the gateway's per-minute cap bit on the first run.
    if (i + BATCH < todo.length) await sleep(1_500);
  }
  return out;
}
