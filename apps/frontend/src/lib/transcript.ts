/**
 * How a sentence from the transcript is shown: in English, with a note when
 * the speaker actually used Hindi.
 *
 * The transcriber writes much of the teacher's English in Devanagari, so the
 * stored `text` is often unreadable to someone who reads neither script; the
 * audio job stores an English `textEn` for every such sentence. The language
 * note rests on the sentence's own Hindi function words (the API's rule), not
 * on its script, so English written in Devanagari is not called Hindi.
 */

export interface DisplayableSentence {
  text: string;
  textEn?: string | null;
  language?: string | null;
}

export function englishOf(s: DisplayableSentence): string {
  return s.textEn?.trim() || s.text;
}

export function languageNote(s: DisplayableSentence, isTeacher: boolean | null): string | null {
  if (s.language !== "hi") return null;
  return isTeacher === false ? "(Hindi)" : "(Teacher used Hindi)";
}

/** "English sentence (Teacher used Hindi)" or just the English. */
export function displayLine(s: DisplayableSentence, isTeacher: boolean | null = true): string {
  const note = languageNote(s, isTeacher);
  return note ? `${englishOf(s)} ${note}` : englishOf(s);
}
