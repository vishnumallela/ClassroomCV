import { describe, expect, test } from "bun:test";
import { needsTranslation, parseTranslations } from "@api/lib/translate";

describe("translate", () => {
  test("only Devanagari sentences are sent", () => {
    expect(needsTranslation("Where is your book?")).toBe(false);
    expect(needsTranslation("आई विल साइन एंड देन रिटर्न।")).toBe(true);
    expect(needsTranslation("Now first we will map the वर्कशीट्स")).toBe(true);
  });

  test("a JSON array reply is read as a whole", () => {
    const t = parseTranslations(
      '[{"i": 1, "en": "I will sign and then return.", "hindi": true}, {"i": 2, "en": "Now we will look at the worksheet.", "hindi": false}]',
    );
    expect(t.get(1)?.en).toBe("I will sign and then return.");
    expect(t.get(2)?.en).toBe("Now we will look at the worksheet.");
  });

  test("the reply is JSON lines; a bad line loses one sentence, not the batch", () => {
    const reply = [
      "```json",
      '{"i": 1, "en": "I will sign and then return.", "hindi": false}',
      "not json at all",
      '{"i": 2, "en": "Do it tomorrow, I did not count yours today.", "hindi": true}',
      '{"i": 3, "en": ""}',
      "```",
    ].join("\n");
    const t = parseTranslations(reply);
    expect(t.get(1)).toEqual({ en: "I will sign and then return.", hindi: false });
    expect(t.get(2)?.hindi).toBe(true);
    expect(t.has(3)).toBe(false);
  });
});
