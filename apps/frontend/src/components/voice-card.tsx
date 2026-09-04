import type { RouterOutputs } from "@classroom/api-contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CircleAlert, Mic } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { msToClock } from "@/lib/format";
import { orpcClient } from "@/lib/orpc";

type Voice = RouterOutputs["videos"]["get"]["voice"];

const LANGUAGE_LABEL: Record<string, string> = { hi: "Hindi", en: "English" };

function pct(share: number): string {
  return `${Math.round(share * 100)}%`;
}

function Tile({
  label,
  value,
  sub,
  muted,
  badge,
}: {
  label: string;
  value: string;
  sub?: string;
  muted?: boolean;
  badge?: string;
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[0.65rem] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        {badge && (
          <Badge variant="secondary" className="text-[0.6rem]">
            {badge}
          </Badge>
        )}
      </div>
      <p
        className={`mt-1 font-display text-xl font-semibold tabular-nums ${muted ? "text-muted-foreground" : ""}`}
      >
        {value}
      </p>
      {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}

/**
 * Group D of the measurements, from the transcript.
 *
 * Every number here is arithmetic over the stored sentences, resolved at read
 * time; the only judgement is whose voice is the teacher's, and that comes
 * from the video's presence timeline, so the card says how sure it is.
 * Questions are provisional until the labelling pass exists, and the card
 * lists what that pass still owes rather than showing zeros for it.
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
  const rerun = useMutation({
    mutationFn: () => orpcClient.analysis.reanalyzeAudio({ id: videoId }),
    onSuccess: () => void queryClient.invalidateQueries(),
  });

  const running =
    voice.audioStatus === "queued" ||
    voice.audioStatus === "extracting" ||
    voice.audioStatus === "transcribing";

  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="flex items-center gap-2 font-display text-base font-semibold tracking-tight">
            <Mic className="size-4 text-muted-foreground" />
            Voice
          </h2>
          <p className="text-xs text-muted-foreground">
            {voice.state === "observed" && voice.teacher.speaker
              ? `Teacher's voice: speaker ${voice.teacher.speaker} (${voice.teacher.confidence}). ${voice.teacher.reason}`
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
        <p className="mt-4 flex items-start gap-1.5 rounded-lg bg-muted/50 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
          <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-tier-medium" />
          <span>Not Observed. {voice.reason}</span>
        </p>
      ) : (
        <>
          {voice.speech && (
            <div className="mt-4 space-y-1.5">
              <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="bg-primary"
                  style={{ width: pct(voice.speech.teacherShare) }}
                  title="Teacher"
                />
                <div
                  className="bg-tier-medium/70"
                  style={{ width: pct(voice.speech.othersShare) }}
                  title="Others"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Teacher {pct(voice.speech.teacherShare)} ({msToClock(voice.speech.teacherMs)}) ·
                others {pct(voice.speech.othersShare)} ({msToClock(voice.speech.othersMs)}) · no
                speech {pct(voice.speech.silenceShare)}
              </p>
            </div>
          )}

          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <Tile
              label="Longest stretch"
              value={voice.longestStretchMs !== null ? msToClock(voice.longestStretchMs) : "—"}
              sub="teacher speaking without a pause"
            />
            <Tile
              label="Pace"
              value={voice.wordsPerMinute !== null ? `${voice.wordsPerMinute}` : "—"}
              sub="words per minute of her speech"
            />
            <Tile
              label="Questions to the class"
              value={voice.questions ? `${voice.questions.toClass}` : "—"}
              sub={
                voice.questions
                  ? `${voice.questions.perTenMinutes} per 10 min · ${voice.questions.checkIns} check-ins ("ठीक है?", "right?") set aside`
                  : undefined
              }
              badge="provisional"
            />
            <Tile
              label="Languages"
              value={
                voice.languages
                  ? voice.languages.shares
                      .map((s) => `${LANGUAGE_LABEL[s.language] ?? s.language} ${pct(s.share)}`)
                      .join(" · ")
                  : "—"
              }
              sub={
                voice.languages
                  ? `${voice.languages.count} used · ${voice.languages.switchesPerMinute} switches per minute of her speech`
                  : undefined
              }
            />
            <Tile
              label="Raised voice"
              value={
                voice.raisedVoice?.state === "observed"
                  ? `${voice.raisedVoice.count}`
                  : "Not Observed"
              }
              sub={
                voice.raisedVoice?.state === "observed"
                  ? `${voice.raisedVoice.perTenMinutes ?? 0} per 10 min · ${voice.raisedVoice.thresholdDb} dB over her own ${voice.raisedVoice.baselineDb} dB baseline`
                  : (voice.raisedVoice?.reason ?? undefined)
              }
              muted={voice.raisedVoice?.state !== "observed"}
            />
            <Tile
              label="Heard"
              value={voice.coverage ? pct(voice.coverage.transcribedShare) : "—"}
              sub={
                voice.coverage
                  ? `${voice.coverage.sentences} sentences, ${voice.coverage.words} words` +
                    (voice.coverage.meanConfidence !== null
                      ? `, ${Math.round(voice.coverage.meanConfidence * 100)}% transcription confidence`
                      : "")
                  : undefined
              }
              muted={!voice.coverage}
            />
          </div>

          {voice.questions && voice.questions.list.length > 0 && (
            <details className="mt-3 rounded-lg border border-border/60 bg-muted/20 p-3">
              <summary className="cursor-pointer text-xs font-medium">
                Her {voice.questions.list.length} questions to the class
              </summary>
              <ol className="mt-2 max-h-64 space-y-1 overflow-y-auto text-xs">
                {voice.questions.list.map((q) => (
                  <li key={q.idx} className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => onSeek(q.atMs)}
                      className="shrink-0 font-mono text-muted-foreground tabular-nums hover:text-foreground"
                    >
                      {msToClock(q.atMs)}
                    </button>
                    <span className="leading-snug">{q.text}</span>
                  </li>
                ))}
              </ol>
            </details>
          )}
        </>
      )}

      <p className="mt-4 text-[0.7rem] leading-relaxed text-muted-foreground">
        Phrase patterns stand in for the labelling pass on: {voice.pendingLabels.join("; ")} — those
        numbers are provisional and show their sentence. A sentence's language is read from its
        Hindi function words, so English the transcriber wrote in Devanagari still counts as
        English.
      </p>
    </Card>
  );
}
