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

export type Language = "hi" | "en";

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
const DEVANAGARI = /[ऀ-ॿ]/u;
const LATIN = /[A-Za-z]/u;
const WORD = /[\p{L}\p{M}\p{N}']+/gu;

// --- which language a sentence is in ---------------------------------------
// The script is NOT the language: the transcriber writes a good deal of the
// teacher's ENGLISH in Devanagari ("आई विल साइन एंड देन रिटर्न"), and reading
// script as language called a lesson 72% Hindi that was taught in English.
// What separates the two is grammar. A Hindi sentence carries Hindi function
// words — postpositions, auxiliaries, pronouns (है, की, को, में, और, नहीं) —
// and transliterated English carries none. So each Devanagari word is scored:
// a Hindi function or common word counts for Hindi, a common English word in
// Devanagari counts for English, and an unknown Devanagari word counts half
// for English, because in these transcripts that is what it usually is.
// Latin-script words are English. Romanised Hindi still counts as English —
// the transcriber does not produce it here, so nothing is lost yet.
const HINDI_WORDS = new Set(
  `है हैं था थे थी हो हूँ हूं हुआ हुई हुए होगा होगी होंगे रहा रही रहे गया गई गए दिया दी दिए लिया ली लिए किया की किये
   करो करे करें करना करता करती करते कर कीजिए कीजिये चाहिए सकता सकती सकते पाएंगे पाओगे करेंगे करोगे देखेंगे पढ़ेंगे लिखेंगे
   जाएंगे आएंगे बताएंगे रहेंगे लेंगे देंगे मिलेगा मिलेगी चलेगा चलेगी
   का के को में से पर और या तो भी ही ना नहीं मत क्या क्यों कैसे कब कहाँ कहां कौन किस किसी कोई कुछ सब सभी बहुत थोड़ा ज़्यादा ज्यादा
   फिर अब अभी आज कल यहाँ यहां वहाँ वहां इधर उधर पहले बाद अंदर बाहर ऊपर नीचे आगे पीछे सामने पास दूर तक साथ बिना जैसे मतलब यानी
   मैं मेरा मेरी मेरे मुझे हम हमारा हमारी हमारे हमें तुम तुम्हारा तुम्हें आप आपका आपकी आपके आपको यह ये वह वो इस उस इन उन इसका उसका
   इसे उसे जो जिस वाला वाली वाले मैंने आपने हमने तुमने उसने इसने किसने अगर लेकिन क्योंकि इसलिए हाँ हां जी
   अच्छा अच्छी ठीक बात लोग बच्चे बच्चों बेटा चलो चलिए देखो देखिए सुनो बोलो बताओ बताइए लो दो दे ले आओ जाओ बैठो उठो लिखो पढ़ो
   समझ समझे समझो अगला अगली अगले प्रश्न सवाल उत्तर जवाब किताब कॉपी पन्ना शब्द वाक्य अर्थ काम पढ़ाई पढ़ना लिखना बोलना
   बड़ा छोटा नया पुराना सारा पूरा आधा एक दो तीन चार पाँच पांच छह सात आठ नौ दस दोस्तों बच्चा लड़का लड़की सर मैम मैडम`
    .split(/\s+/u)
    .filter(Boolean),
);
const ENGLISH_IN_DEVANAGARI = new Set(
  `आई यू वी दे द इज़ इज आर वाज़ वाज वर बीन टू तू एंड नॉट विल हैव हैज़ हैज डू डज़ डिड डन इट दिस दैट दीज़ देयर हियर नाउ ओके ओकै
   प्लीज गो कम सो बट ऑफ़ ऑफ फॉर इन ऑन विद व्हाट व्हेन व्हेयर हू व्हाई हाउ कैन शुड मस्ट यस नो ओनली आल्सो वेरी जस्ट लाइक
   वन थ्री फर्स्ट सेकंड नेक्स्ट लास्ट टुडे टुमारो टुमरो टुमॉरो चेक राइट रीड ओपन क्लोज़ क्लोज बुक बुक्स पेज वर्कशीट वर्कशीट्स
   क्लास टीचर स्टूडेंट होमवर्क क्वेश्चन आंसर सेंटेंस वर्ड मीनिंग टाइम मिनट देन दैन बिफोर आफ्टर अगेन थिंग समथिंग एवरीथिंग
   योर माई अस हिम हर देम इफ बिकॉज़ बिकॉज व्हिच वेट स्टार्ट फिनिश कंप्लीट सबमिट ब्रिंग टेक पुट कीप गिव गेट टेल से सेड लिसन
   लुक राइटिंग रीडिंग टिक क्रॉस कंटेंट नंबर पार्ट फुल हाफ गुड बैड न्यू ओल्ड बिग स्मॉल फास्ट स्लो`
    .split(/\s+/u)
    .filter(Boolean),
);
const HINDI_SUFFIX = /(ेंगे|ेंगी|ेगा|ेगी|ोगे|ोगी|ूँगा|ूंगा|ूँगी|ूंगी|िए|िये|ाइए|ाइये)$/u;

/** How much of a sentence is Hindi and how much English, as word evidence. */
export function languageMix(text: string): { hi: number; en: number } {
  let hi = 0;
  let en = 0;
  for (const raw of text.match(WORD) ?? []) {
    const w = raw.replace(/[।॥]/gu, "");
    if (!w) continue;
    if (LATIN.test(w)) en += 1;
    else if (DEVANAGARI.test(w)) {
      if (HINDI_WORDS.has(w) || HINDI_SUFFIX.test(w)) hi += 1;
      else if (ENGLISH_IN_DEVANAGARI.has(w)) en += 1;
      else en += 0.5;
    }
  }
  return { hi, en };
}

/** The language a sentence is mostly in; ties go to Hindi, since a Hindi
 *  frame with English nouns in it is Hindi. Null when it has no words. */
export function languageOf(text: string): Language | null {
  const { hi, en } = languageMix(text);
  if (hi === 0 && en === 0) return null;
  return hi >= en ? "hi" : "en";
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
