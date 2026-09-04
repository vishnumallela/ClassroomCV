import { describe, expect, test } from "bun:test";
import {
  languageMix,
  languageOf,
  MAX_WORDS,
  PAUSE_MS,
  segmentSentences,
  type Word,
} from "@api/lib/segment";

function words(spec: [string, number, number, string?][], speaker = "B"): Word[] {
  return spec.map(([text, start, end, sp]) => ({
    text,
    start,
    end,
    confidence: 0.9,
    speaker: sp ?? speaker,
  }));
}

describe("languageOf", () => {
  test("Hindi is its function words, not its script", () => {
    expect(languageOf("अब हम वर्कशीट देखेंगे।")).toBe("hi");
    expect(languageOf("आप कल कर लेना, मैंने आज आपका नहीं माना।")).toBe("hi");
    expect(languageOf("Take out your literacy companion.")).toBe("en");
    // English written in Devanagari by the transcriber is English
    expect(languageOf("आई विल साइन एंड देन रिटर्न।")).toBe("en");
    expect(languageOf("वर्कशीट नंबर फोर्टी वन नाउ,")).toBe("en");
    expect(languageOf("...")).toBeNull();
  });

  test("a Hindi frame with English nouns is Hindi; an English frame with Hindi words is English", () => {
    expect(languageOf("अब worksheet का page open करो")).toBe("hi");
    expect(languageOf("आज आई आज हम चेकिंग इट हियर।")).toBe("en");
    const mix = languageMix("अब worksheet का page open करो");
    expect(mix.hi).toBe(3);
    expect(mix.en).toBe(3);
  });
});

describe("segmentSentences", () => {
  test("a turn is cut at sentence-final punctuation, including the danda", () => {
    const s = segmentSentences(
      words([
        ["Open", 0, 300],
        ["your", 300, 500],
        ["books.", 500, 900],
        ["अब", 1000, 1200],
        ["पढ़ो।", 1200, 1600],
        ["Who", 1700, 1900],
        ["knows?", 1900, 2300],
      ]),
    );
    expect(s.map((x) => x.text)).toEqual(["Open your books.", "अब पढ़ो।", "Who knows?"]);
    expect(s[0]).toMatchObject({ start: 0, end: 900, speaker: "B", language: "en" });
    expect(s[1]?.language).toBe("hi");
  });

  test("a pause ends a sentence even without punctuation", () => {
    const s = segmentSentences(
      words([
        ["so", 0, 200],
        ["then", 200, 500],
        ["next", 500 + PAUSE_MS, 800 + PAUSE_MS],
        ["one", 800 + PAUSE_MS, 1000 + PAUSE_MS],
      ]),
    );
    expect(s.map((x) => x.text)).toEqual(["so then", "next one"]);
  });

  test("a speaker change ends a sentence mid-punctuation", () => {
    const s = segmentSentences(
      words([
        ["Twelve", 0, 300, "A"],
        ["Margav", 350, 700, "B"],
        ["Give", 750, 900, "A"],
        ["it", 900, 1000, "A"],
      ]),
    );
    expect(s.map((x) => [x.speaker, x.text])).toEqual([
      ["A", "Twelve"],
      ["B", "Margav"],
      ["A", "Give it"],
    ]);
  });

  test("an unpunctuated run is capped so one row cannot swallow a lecture", () => {
    const spec: [string, number, number][] = [];
    for (let i = 0; i < 150; i++) spec.push([`w${i}`, i * 200, i * 200 + 150]);
    const s = segmentSentences(words(spec));
    expect(s.length).toBeGreaterThanOrEqual(3);
    for (const x of s) expect(x.text.split(" ").length).toBeLessThanOrEqual(MAX_WORDS);
  });

  test("confidence is the mean of the words", () => {
    const w = words([
      ["a", 0, 100],
      ["b.", 100, 200],
    ]);
    w[0]!.confidence = 0.5;
    w[1]!.confidence = 1.0;
    expect(segmentSentences(w)[0]?.confidence).toBeCloseTo(0.75);
  });

  test("no words, no sentences", () => {
    expect(segmentSentences([])).toEqual([]);
  });
});
