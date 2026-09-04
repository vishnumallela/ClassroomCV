import type { RouterOutputs } from "@classroom/api-contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Stat } from "@/components/ui/stat";
import { msToClock } from "@/lib/format";
import { percent } from "@/lib/measures";
import { orpcClient } from "@/lib/orpc";
import { displayLine } from "@/lib/transcript";

type Voice = RouterOutputs["videos"]["get"]["voice"];

const LANGUAGE_LABEL: Record<string, string> = { hi: "Hindi", en: "English" };

/**
 * What the microphone heard: who spoke how much, how she spoke, and what
 * she asked. Every number is arithmetic over the stored sentences; the one
 * judgement — whose voice is hers — comes from the video and is stated with
 * its confidence. The sentences sit behind toggles so the numbers read first.
 */
export function VoiceCard({
  videoId,
  voice,
  onSeek,
}: {
  videoId: string;
  voice: Voice;
  onSeek: (ms: number) => void;
}) {
  const queryClient = useQueryClient();
  const [showQuestions, setShowQuestions] = useState(false);
  const [showMethod, setShowMethod] = useState(false);
  const rerun = useMutation({
    mutationFn: () => orpcClient.analysis.reanalyzeAudio({ id: videoId }),
    onSuccess: () => void queryClient.invalidateQueries(),
  });
  const running =
    voice.audioStatus === "queued" ||
    voice.audioStatus === "extracting" ||
    voice.audioStatus === "transcribing";

  const languages = voice.languages
    ? voice.languages.shares
        .map((s) => `${LANGUAGE_LABEL[s.language] ?? s.language} ${percent(s.share)}`)
        .join(" · ")
    : "—";
  const languageSub = voice.languages
    ? voice.languages.teacherUsedHindi
      ? `The teacher used Hindi during the lesson (${voice.languages.hindiSentences} sentence${voice.languages.hindiSentences === 1 ? "" : "s"})`
      : "The teacher taught in English"
    : null;

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-display text-base font-semibold tracking-tight">Voice</h2>
          <p className="text-xs text-muted-foreground">
            {voice.state === "observed" && voice.teacher.speaker
              ? `Her voice is speaker ${voice.teacher.speaker} (${voice.teacher.confidence} confidence): it carried most of the speech while the video shows her in the room.`
              : "From the microphone, once the transcript exists."}
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          disabled={running || rerun.isPending}
          onClick={() => rerun.mutate()}
        >
          {running ? "Running…" : "Re-run audio"}
        </Button>
      </div>

      {voice.state !== "observed" ? (
        <p className="mt-4 rounded-lg bg-muted/50 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
          Not observed. {voice.reason}
        </p>
      ) : (
        <>
          {voice.speech && (
            <div className="mt-4 space-y-1.5">
              <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="bg-primary"
                  style={{ width: percent(voice.speech.teacherShare) }}
                  title="Teacher"
                />
                <div
                  className="bg-tier-medium/70"
                  style={{ width: percent(voice.speech.othersShare) }}
                  title="Others"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Teacher spoke {percent(voice.speech.teacherShare)} of the time (
                {msToClock(voice.speech.teacherMs)}), others {percent(voice.speech.othersShare)},
                nobody {percent(voice.speech.silenceShare)}
              </p>
            </div>
          )}

          <div className="mt-5 grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
            <Stat
              label="Longest stretch"
              value={voice.longestStretchMs !== null ? msToClock(voice.longestStretchMs) : "—"}
              sub="teacher speaking without a pause"
              state="observed"
            />
            <Stat
              label="Pace"
              value={voice.wordsPerMinute !== null ? `${voice.wordsPerMinute} wpm` : "—"}
              sub="words per minute of her speech"
              state="observed"
            />
            <Stat
              id="R20"
              label="Questions to the class"
              value={voice.questions ? `${voice.questions.toClass}` : "—"}
              sub={
                voice.questions
                  ? `${voice.questions.perTenMinutes} per 10 min · ${voice.questions.checkIns} check-ins like “ठीक है?” set aside`
                  : null
              }
              state="provisional"
              reason="Counted from question marks until the labelling pass exists."
            />
            <Stat id="R21" label="Languages" value={languages} sub={languageSub} state="observed" />
            <Stat
              id="R17"
              label="Raised voice"
              value={voice.raisedVoice?.state === "observed" ? `${voice.raisedVoice.count}` : "—"}
              sub={
                voice.raisedVoice?.state === "observed"
                  ? `${voice.raisedVoice.perTenMinutes ?? 0} per 10 min · ${voice.raisedVoice.thresholdDb} dB over her own baseline`
                  : null
              }
              state={voice.raisedVoice?.state === "observed" ? "observed" : "not_observed"}
              reason={voice.raisedVoice?.reason}
            />
            <Stat
              id="R22"
              label="Heard"
              value={voice.coverage ? percent(voice.coverage.transcribedShare) : "—"}
              sub={
                voice.coverage
                  ? `${voice.coverage.sentences} sentences · ${Math.round((voice.coverage.meanConfidence ?? 0) * 100)}% transcription confidence`
                  : null
              }
              state="observed"
            />
          </div>

          <div className="mt-4 flex flex-wrap gap-4 text-xs">
            {voice.questions && voice.questions.list.length > 0 && (
              <button
                type="button"
                onClick={() => setShowQuestions((v) => !v)}
                className="text-muted-foreground underline-offset-2 hover:underline"
              >
                {showQuestions
                  ? "Hide her questions"
                  : `Show her ${voice.questions.list.length} questions`}
              </button>
            )}
            <button
              type="button"
              onClick={() => setShowMethod((v) => !v)}
              className="text-muted-foreground underline-offset-2 hover:underline"
            >
              {showMethod ? "Hide how these are measured" : "How these are measured"}
            </button>
          </div>

          {showQuestions && voice.questions && (
            <ol className="mt-3 max-h-64 space-y-1 overflow-y-auto rounded-lg border border-border/60 bg-muted/20 p-3 text-xs">
              {voice.questions.list.map((q) => (
                <li key={q.idx} className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => onSeek(q.atMs)}
                    className="shrink-0 font-mono text-muted-foreground tabular-nums hover:text-foreground"
                  >
                    {msToClock(q.atMs)}
                  </button>
                  <span className="leading-snug" title={q.text}>
                    {displayLine(q, true)}
                  </span>
                </li>
              ))}
            </ol>
          )}
        </>
      )}

      {showMethod && (
        <p className="mt-3 text-[0.7rem] leading-relaxed text-muted-foreground">
          Sentences are cut from the transcript's word timings and shown in English; where the
          original was Hindi, the line says so, and English the transcriber wrote in Devanagari
          counts as English. Questions are counted from question marks with check-ins set aside.
          Raised voice is her sentences {voice.raisedVoice?.thresholdDb ?? 6} dB over her own median
          loudness for at least 1.5 s. Phrase patterns stand in for the labelling pass on:{" "}
          {voice.pendingLabels.join("; ")}.
        </p>
      )}
    </Card>
  );
}
