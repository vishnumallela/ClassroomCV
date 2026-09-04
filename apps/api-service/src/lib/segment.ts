/**
 * Cutting a transcript into sentences.
 *
 * The diarizer returns speaker TURNS: everything one person said until
 * somebody else spoke. On a real lesson that is the wrong unit — the first
 * transcript came back as 52 turns for 45 minutes, ten of them longer than a
 * minute and one 6.5 minutes and 3,650 characters long, because a teacher
 * lecturing is one turn. Every voice measurement (questions, languages,
 * switches, the labelling pass) is per utterance, so a turn that long hides
 * 98 pauses and dozens of sentences inside one row.
 *
 * The words carry exact timestamps and the speaker, so the cut is made on
 * them: at sentence-final punctuation, at a pause longer than PAUSE_MS, at a
 * speaker change, and at a cap so an unpunctuated run cannot become one row.
 * Pure, so it can be tested without the provider.
 */

export interface Word {
  text: string;
  start: number;
  end: number;
  confidence: number;
  speaker: string | null;
}

export type Language = "hi" | "en" | "mixed";

export interface Sentence {
  speaker: string;
  start: number;
  end: number;
  text: string;
  confidence: number;
  language: Language | null;
}

/** A gap between words this long ends a sentence whatever the punctuation. */
export const PAUSE_MS = 1_000;
/** No sentence runs past either of these, punctuation or not. */
export const MAX_WORDS = 60;
export const MAX_MS = 25_000;

const SENTENCE_END = /[.?!।॥]["'”’)]*$/u;
const DEVANAGARI = /[ऀ-ॿ]/gu;
const LATIN = /[A-Za-z]/gu;

/**
 * Which language a sentence is in, read from its script: the provider returns
 * Hindi in Devanagari and English in Latin letters, and no per-word language
 * field. Romanised Hindi therefore counts as English — a known limit that a
 * transliteration pass would lift. "mixed" when neither script dominates.
 */
export function languageOf(text: string): Language | null {
  const hi = (text.match(DEVANAGARI) ?? []).length;
  const en = (text.match(LATIN) ?? []).length;
  if (hi === 0 && en === 0) return null;
  if (en === 0 || hi >= 4 * en) return "hi";
  if (hi === 0 || en >= 4 * hi) return "en";
  return "mixed";
}

export function segmentSentences(words: Word[]): Sentence[] {
  const out: Sentence[] = [];
  let cur: Word[] = [];

  const flush = () => {
    if (cur.length === 0) return;
    const first = cur[0];
    const last = cur[cur.length - 1];
    if (!first || !last) return;
    const text = cur.map((w) => w.text).join(" ");
    out.push({
      speaker: first.speaker ?? "?",
      start: first.start,
      end: last.end,
      text,
      confidence: cur.reduce((s, w) => s + w.confidence, 0) / cur.length,
      language: languageOf(text),
    });
    cur = [];
  };

  for (const w of words) {
    const prev = cur[cur.length - 1];
    if (prev) {
      const speakerChanged = (w.speaker ?? "?") !== (prev.speaker ?? "?");
      const paused = w.start - prev.end >= PAUSE_MS;
      const ended = SENTENCE_END.test(prev.text);
      const first = cur[0];
      const capped =
        cur.length >= MAX_WORDS || (first !== undefined && w.end - first.start > MAX_MS);
      if (speakerChanged || paused || ended || capped) flush();
    }
    cur.push(w);
  }
  flush();
  return out;
}
