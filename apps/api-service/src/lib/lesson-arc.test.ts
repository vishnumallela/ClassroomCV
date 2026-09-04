import { describe, expect, test } from "bun:test";
import { lessonArc, type ArcSentence } from "@api/lib/lesson-arc";
import { parseAstats, raisedVoiceEvents, sentenceLoudness } from "@api/lib/loudness";
import { labelSentence } from "@api/lib/phrases";

let n = 0;
function s(startMs: number, endMs: number, text: string): ArcSentence {
  return { idx: n++, startMs, endMs, text };
}

describe("labelSentence (phrase tables, both scripts)", () => {
  test("task launches, in English and in Devanagari-rendered English", () => {
    expect(
      labelSentence("By the time ma'am is collecting, you take out your literacy companion.")
        .setsTask,
    ).toBe(true);
    expect(labelSentence("वर्कशीट नंबर फोर्टी वन नाउ,").setsTask).toBe(true);
    expect(labelSentence("Satwik, now let us start one and eighteen.").setsTask).toBe(true);
    expect(labelSentence("This is serious business.").setsTask).toBe(false);
  });

  test("attention cues", () => {
    expect(labelSentence("By the count of 5, you need to quieten up.").attentionCue).toBe(true);
    expect(labelSentence("Goran, you are not listening.").attentionCue).toBe(true);
    expect(labelSentence("सुनो सब लोग, इधर देखो।").attentionCue).toBe(true);
    expect(labelSentence("Where is your book?").attentionCue).toBe(false);
  });

  test("homework and continuation, including the transliterated forms", () => {
    expect(labelSentence("टुमारो यू आर गेटिंग अ होमवर्क इफ यू हैव नॉट डन इट।").homework).toBe(true);
    expect(labelSentence("यू डू, ब्रिंग इट टुमारो।").homework).toBe(true);
    expect(labelSentence("We will continue this next class.").continuation).toBe(true);
    expect(labelSentence("बाकी कल करेंगे।").continuation).toBe(true);
    expect(labelSentence("Read the sentence and find the meaning.").homework).toBe(false);
  });

  test("pack-up needs an imperative: a dictated sentence about bags is not one", () => {
    expect(labelSentence("Pack up your bags and line up.").packUp).toBe(true);
    expect(labelSentence("किताबें बंद करो, बैग पैक करो।").packUp).toBe(true);
    expect(labelSentence("द बॉयज़ बैग्स वेर केप्ट आउटसाइड।").packUp).toBe(false);
  });

  test("closure types", () => {
    expect(labelSentence("Let's quickly recap what we learnt today.").closure).toBe("review");
    expect(labelSentence("Before you go, one last question.").closure).toBe("exit_question");
    expect(labelSentence("Finish it fast, Namish.").closure).toBeNull();
  });

  test("procedure", () => {
    expect(labelSentence("Who is roll number twelve? Give the notebook here.").procedure).toBe(
      true,
    );
    expect(labelSentence("गेट साइन हरिया फिनिश इट").procedure).toBe(true);
  });
});

describe("lessonArc", () => {
  // The real lesson's shape in miniature. Bells at 2 s and 2702 s; she arrives
  // at 247 s and leaves at 2700 s; board 255-336 s and 2507-2672 s.
  const base = {
    boardIntervals: [
      [255_000, 336_000],
      [2_507_000, 2_672_000],
    ] as [number, number][],
    durationMs: 2_700_922,
    bellStartMs: 2_000,
    bellEndMs: 2_702_000,
    arrivalMs: 246_728,
    departureMs: 2_700_802,
  };
  const sentences = [
    s(150_000, 155_000, "Who is roll number twelve? Give the notebook here."),
    s(289_000, 296_000, "By the time ma'am is collecting, you take out your literacy companion."),
    s(
      375_000,
      384_000,
      "By the count of 5, you need to quieten up and take out your literacy companion.",
    ),
    s(1_620_000, 1_623_000, "Goran, you are not listening."),
    s(2_332_000, 2_336_000, "टुमारो यू आर गेटिंग अ होमवर्क इफ यू हैव नॉट डन इट।"),
    s(2_567_000, 2_571_000, "When there are two sentences, you will have to put semicolon."),
    s(2_605_000, 2_608_000, "यू डू, ब्रिंग इट टुमारो।"),
    s(2_651_000, 2_654_000, "गेट साइन हरिया फिनिश इट"),
    s(2_695_000, 2_698_000, "हु इज़ द मॉनिटर ऑफ़ दिस बोर्ड?"),
  ];

  test("start is the first task-setting sentence after her arrival, corroborated by the board", () => {
    const arc = lessonArc({ ...base, sentences });
    expect(arc.start.value).toBe(289_000);
    expect(arc.start.state).toBe("provisional");
    expect(arc.start.corroboratedByBoard).toBe(true);
    expect(arc.start.evidence[0]?.text).toContain("take out your literacy companion");
    expect(arc.startDelayMin.value).toBe(4.8);
  });

  test("end is the later of the last teaching sentence and leaving the board, capped at departure", () => {
    const arc = lessonArc({ ...base, sentences });
    // last teaching sentence ends 2571 s; board left at 2672 s; departure 2700.8 s
    expect(arc.end.value).toBe(2_672_000);
    expect(arc.durationMin.value).toBe(39.7);
    expect(arc.fitsPeriod.value).toBe(true);
    expect(arc.overrunMin.value).toBe(-0.5);
  });

  test("the ending: no closure found, homework yes, continuation no", () => {
    const arc = lessonArc({ ...base, sentences });
    expect(arc.closure.value).toBe("none");
    expect(arc.homework.value).toBe(true);
    expect(arc.homework.atMs).toBe(2_332_000);
    expect(arc.continuation.value).toBe(false);
  });

  test("attention requests and administrative drift", () => {
    const arc = lessonArc({ ...base, sentences });
    expect(arc.attentionRequests.value).toBe(2);
    // 2651 s and 2695 s are 44 s apart: two lone procedural sentences, not a run
    expect(arc.drift.value).toEqual({ episodes: 0, totalMs: 0 });
    // ...but procedural sentences within thirty seconds of each other are one
    // episode, however the rows arrive (the 2695 s "monitor" sentence joins it)
    const withRun = lessonArc({
      ...base,
      sentences: [
        ...sentences,
        s(2_670_000, 2_673_000, "Submit your planners to the monitor."),
        s(2_680_000, 2_683_000, "Sign the diary as well."),
      ],
    });
    expect(withRun.drift.value).toEqual({ episodes: 1, totalMs: 2_698_000 - 2_670_000 });
    expect(withRun.drift.evidence[0]?.atMs).toBe(2_680_000);
  });

  test("a recap in the last five minutes names the closure", () => {
    const arc = lessonArc({
      ...base,
      sentences: [
        ...sentences,
        s(2_660_000, 2_665_000, "Let's quickly recap what we learnt today."),
      ],
    });
    expect(arc.closure.value).toBe("review");
    expect(arc.closure.evidence[0]?.atMs).toBe(2_660_000);
  });

  test("no transcript: start falls back to the board, the rest is Not Observed with a reason", () => {
    const arc = lessonArc({ ...base, sentences: [] });
    expect(arc.start.value).toBe(255_000);
    expect(arc.start.reason).toContain("first board interaction");
    expect(arc.closure.state).toBe("not_observed");
    expect(arc.homework.state).toBe("not_observed");
    expect(arc.attentionRequests.state).toBe("not_observed");
  });

  test("no bells: the delays are Not Observed, the times still stand", () => {
    const arc = lessonArc({ ...base, sentences, bellStartMs: null, bellEndMs: null });
    expect(arc.start.value).toBe(289_000);
    expect(arc.startDelayMin.state).toBe("not_observed");
    expect(arc.fitsPeriod.state).toBe("not_observed");
    expect(arc.overrunMin.state).toBe("not_observed");
  });
});

describe("loudness", () => {
  test("parses ffmpeg's metadata print", () => {
    const w = parseAstats(
      "frame:0    pts:0       pts_time:0\nlavfi.astats.Overall.RMS_level=-14.5\n" +
        "frame:1    pts:8000    pts_time:0.5\nlavfi.astats.Overall.RMS_level=-inf\n" +
        "frame:2    pts:16000   pts_time:1\nlavfi.astats.Overall.RMS_level=-23.4\n",
    );
    expect(w).toEqual([
      { t: 0, rms: -14.5 },
      { t: 0.5, rms: -120 },
      { t: 1, rms: -23.4 },
    ]);
  });

  test("a sentence's loudness is the power mean of its voiced windows and its peak", () => {
    const w = [
      { t: 0, rms: -20 },
      { t: 0.5, rms: -20 },
      { t: 1, rms: -80 }, // silence, ignored
      { t: 1.5, rms: -14 },
    ];
    const l = sentenceLoudness(w, 0, 2_000);
    expect(l.peakDb).toBe(-14);
    expect(l.rmsDb).toBeGreaterThan(-20);
    expect(l.rmsDb).toBeLessThan(-14);
    expect(sentenceLoudness(w, 5_000, 6_000)).toEqual({ rmsDb: null, peakDb: null });
  });

  test("raised voice: sentences well above her own median, long enough, merged into episodes", () => {
    const quiet = Array.from({ length: 12 }, (_, i) => ({
      startMs: i * 10_000,
      endMs: i * 10_000 + 3_000,
      rmsDb: -22,
    }));
    const loud = [
      { startMs: 200_000, endMs: 203_000, rmsDb: -14 },
      { startMs: 204_000, endMs: 207_000, rmsDb: -15 }, // 1 s later: same episode
      { startMs: 300_000, endMs: 300_800, rmsDb: -10 }, // too short
      { startMs: 400_000, endMs: 403_000, rmsDb: -17 }, // only 5 dB above
    ];
    const r = raisedVoiceEvents([...quiet, ...loud]);
    expect(r.baselineDb).toBe(-22);
    expect(r.events).toHaveLength(1);
    expect(r.events[0]).toMatchObject({ startMs: 200_000, endMs: 207_000, dbAbove: 8 });
  });

  test("too few sentences to know her baseline: no events, no baseline", () => {
    expect(raisedVoiceEvents([{ startMs: 0, endMs: 3_000, rmsDb: -10 }])).toEqual({
      events: [],
      baselineDb: null,
    });
  });
});
