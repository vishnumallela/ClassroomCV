import { beforeEach, describe, expect, mock, test } from "bun:test";

/**
 * What gets sent to AssemblyAI, checked without sending it.
 *
 * The bug this file exists for cost a real run: the body carried both
 * `language_codes` and `language_detection`, which the API rejects outright
 * ("`language_detection` is not available when `language_codes` is
 * specified"). Nothing was billed — it fails before upload — but the audio
 * pipeline sat broken, and the only signal was a 400 nobody was watching for.
 *
 * Every assertion below is a pairwise rule the API enforces at submit, or a
 * setting whose ABSENCE is silent rather than loud: omit `speech_models` and
 * the account default answers instead, returning one speaker and ignoring the
 * key terms, with a 200 and a plausible-looking transcript. That failure mode
 * is why these are pinned here rather than left to the first real lesson.
 */

mock.module("@api/lib/app-settings", () => ({
  getAppSettings: () => Promise.resolve({ assemblyaiApiKey: "test-key" }),
}));

type Body = Record<string, unknown>;

/** Submit once against a stubbed fetch and return the parsed request body. */
async function submittedBody(options?: { keyTerms?: string[]; prompt?: string }): Promise<Body> {
  const { submitTranscript } = await import("@api/lib/transcribe");
  let captured: Body | undefined;

  globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
    const href = String(url);
    if (href.endsWith("/upload")) {
      return Promise.resolve(
        new Response(JSON.stringify({ upload_url: "https://cdn.test/audio.flac" })),
      );
    }
    captured = JSON.parse(String(init?.body)) as Body;
    return Promise.resolve(new Response(JSON.stringify({ id: "transcript-1" })));
  }) as typeof fetch;

  await submitTranscript("/nonexistent/audio.flac", options ?? {});
  if (!captured) throw new Error("no transcript request was made");
  return captured;
}

describe("submitTranscript request body", () => {
  let body: Body;
  beforeEach(async () => {
    body = await submittedBody();
  });

  test("never sends language_detection alongside language_codes", () => {
    // The regression. These two are mutually exclusive at submit.
    expect(body).not.toHaveProperty("language_detection");
    expect(body.language_codes).toEqual(["en", "hi"]);
  });

  test("keeps code switching rather than dominant-language detection", () => {
    // Resolving the conflict the other way would have been worse than the bug:
    // language_detection picks ONE language for the file, so a Hinglish lesson
    // loses whichever half is not dominant — and it reports no per-utterance
    // language either, so it buys nothing in exchange.
    expect(body.language_codes).toHaveLength(2);
    // The API requires that one of the two codes be en.
    expect(body.language_codes).toContain("en");
  });

  test("names the speech model explicitly, plural and as an array", () => {
    // `speech_model` (singular) is deprecated, and omitting it entirely is the
    // silent failure: the account default runs and ignores the key terms.
    expect(body.speech_models).toEqual(["universal-3-5-pro"]);
    expect(body).not.toHaveProperty("speech_model");
  });

  test("gives the diarizer a speaker floor, and not the conflicting field", () => {
    // speaker_labels alone returned a 4.5-minute lesson as ONE speaker.
    expect(body.speaker_labels).toBe(true);
    expect(body.speaker_options).toEqual({
      min_speakers_expected: 2,
      max_speakers_expected: 6,
    });
    // speaker_options and speakers_expected cannot both be sent.
    expect(body).not.toHaveProperty("speakers_expected");
  });

  test("keeps disfluencies, which are a graded behaviour here", () => {
    expect(body.disfluencies).toBe(true);
  });
});

describe("submitTranscript vocabulary", () => {
  test("sends key terms, never alongside a prompt", async () => {
    const body = await submittedBody({ prompt: "a classroom lesson" });
    // The API rejects a request carrying both; key terms win because the
    // measurements are found BY those phrases.
    expect(Array.isArray(body.keyterms_prompt)).toBe(true);
    expect(body).not.toHaveProperty("prompt");
  });

  test("falls back to the prompt only when there are no key terms", async () => {
    const body = await submittedBody({ keyTerms: [] });
    // The built-in classroom terms are always present, so key terms never empty
    // in practice — this pins that the built-ins are actually sent.
    expect(body.keyterms_prompt).toContain("ma'am");
    expect(body).not.toHaveProperty("prompt");
  });
});
