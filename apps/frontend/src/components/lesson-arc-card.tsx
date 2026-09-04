import type { RouterOutputs } from "@classroom/api-contracts";
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Stat, StateChip, type MeasureState } from "@/components/ui/stat";
import { msToClock } from "@/lib/format";
import { againstBell, clockAt } from "@/lib/measures";
import { displayLine } from "@/lib/transcript";

type Arc = RouterOutputs["videos"]["get"]["arc"];
type Evidence = {
  idx: number;
  atMs: number;
  text: string;
  textEn?: string | null;
  language?: string | null;
};

const CLOSURE_LABEL: Record<string, string> = {
  review: "Review",
  reflection: "Reflection",
  exit_question: "Exit question",
  summary: "Summary",
  none: "None",
};

/**
 * The lesson itself — when teaching started and ended, how it ended, and how
 * the class was managed — as one card of plain-language stats. Every number
 * carries its state; the sentences each rests on sit behind one toggle so the
 * numbers can be read first and checked second (a time seeks the video).
 */
export function LessonArcCard({
  arc,
  recordingStartedAt,
  timezone,
  onSeek,
}: {
  arc: Arc;
  recordingStartedAt: string | null;
  timezone: string;
  onSeek: (ms: number) => void;
}) {
  const [showEvidence, setShowEvidence] = useState(false);
  const measures = [
    arc.start,
    arc.startDelayMin,
    arc.end,
    arc.durationMin,
    arc.fitsPeriod,
    arc.overrunMin,
    arc.closure,
    arc.continuation,
    arc.homework,
    arc.attentionRequests,
    arc.drift,
  ];
  const allProvisional = measures.every((m) => m.state === "provisional");
  // When the whole card is provisional, say it once in the header instead of
  // on every number; a stat still shows its own chip when it differs.
  const chip = (state: string): MeasureState | undefined =>
    allProvisional && state === "provisional" ? undefined : (state as MeasureState);
  const at = (ms: number | null) => {
    if (ms === null) return "—";
    return clockAt(recordingStartedAt, ms, timezone) ?? msToClock(ms);
  };
  const st = chip;

  const startSub =
    arc.start.value === null
      ? ""
      : arc.start.corroborated
        ? "her words and the board agree"
        : arc.start.actionMs !== null && arc.start.voiceMs === null
          ? "from writing or pointing at the board"
          : "from her words";
  const endSub =
    arc.end.value === null
      ? ""
      : arc.end.corroboratedByBoard
        ? "leaving the board confirms it"
        : "from her last teaching sentence";

  const evidence: { title: string; items: Evidence[] }[] = [
    { title: "Teaching started", items: arc.start.evidence },
    { title: "Teaching ended", items: arc.end.evidence },
    { title: "How it ended", items: arc.closure.evidence },
    { title: "Continues next time", items: arc.continuation.evidence },
    { title: "Homework", items: arc.homework.evidence },
    { title: "Attention requests", items: arc.attentionRequests.evidence },
    { title: "Off-lesson talk", items: arc.drift.evidence },
  ].filter((g) => g.items.length > 0);

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="flex items-center gap-2 font-display text-base font-semibold tracking-tight">
            The lesson
            {allProvisional && (
              <StateChip
                state="provisional"
                title="Read from phrase patterns until the labelling pass exists; the sentences are behind the toggle."
              />
            )}
          </h2>
          <p className="text-xs text-muted-foreground">
            Start from her first task-setting words or her first writing or pointing at the board;
            end from her last teaching words or the last time she left the board.
            {allProvisional && " Provisional: read by phrase until the labelling pass exists."}
          </p>
        </div>
        {evidence.length > 0 && (
          <button
            type="button"
            onClick={() => setShowEvidence((v) => !v)}
            className="text-xs text-muted-foreground underline-offset-2 hover:underline"
          >
            {showEvidence ? "Hide the sentences" : "Show the sentences behind these"}
          </button>
        )}
      </div>

      <div className="mt-4 grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
        <Stat
          id="R7"
          label="Teaching started"
          value={at(arc.start.value)}
          sub={startSub}
          state={st(arc.start.state)}
          reason={arc.start.reason}
        />
        <Stat
          id="R8"
          label="Start delay"
          value={againstBell(arc.startDelayMin.value, "after the bell", "before the bell")}
          state={st(arc.startDelayMin.state)}
          reason={arc.startDelayMin.reason}
        />
        <Stat
          id="R9"
          label="Teaching ended"
          value={at(arc.end.value)}
          sub={endSub}
          state={st(arc.end.state)}
          reason={arc.end.reason}
        />
        <Stat
          id="R10"
          label="Taught for"
          value={arc.durationMin.value !== null ? `${arc.durationMin.value} min` : "—"}
          state={st(arc.durationMin.state)}
          reason={arc.durationMin.reason}
        />
        <Stat
          id="R11"
          label="Fit the period"
          value={arc.fitsPeriod.value === null ? "—" : arc.fitsPeriod.value ? "Yes" : "No"}
          state={st(arc.fitsPeriod.state)}
          reason={arc.fitsPeriod.reason}
        />
        <Stat
          id="R12"
          label="Over- or under-run"
          value={againstBell(
            arc.overrunMin.value,
            "past the end bell",
            "of the period unused",
            "on the bell",
          )}
          state={st(arc.overrunMin.state)}
          reason={arc.overrunMin.reason}
        />
      </div>

      <div className="mt-6 grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
        <Stat
          id="R13"
          label="How it ended"
          value={arc.closure.value ? (CLOSURE_LABEL[arc.closure.value] ?? arc.closure.value) : "—"}
          sub={
            arc.closure.value === "none" ? "no review, reflection, exit question or summary" : null
          }
          state={st(arc.closure.state)}
          reason={arc.closure.reason}
        />
        <Stat
          id="R14"
          label="Continues next time"
          value={
            arc.continuation.value === null
              ? "—"
              : arc.continuation.value
                ? "Yes, she said so"
                : "Not said"
          }
          state={st(arc.continuation.state)}
          reason={arc.continuation.reason}
        />
        <Stat
          id="R15"
          label="Homework set"
          value={
            arc.homework.value === null
              ? "—"
              : arc.homework.value
                ? `Yes, at ${at(arc.homework.atMs)}`
                : "No"
          }
          state={st(arc.homework.state)}
          reason={arc.homework.reason}
        />
        <Stat
          id="R18"
          label="Attention requests"
          value={arc.attentionRequests.value !== null ? `${arc.attentionRequests.value}` : "—"}
          sub={
            arc.attentionRequests.perTenMinutes !== null
              ? `${arc.attentionRequests.perTenMinutes} per 10 min`
              : null
          }
          state={st(arc.attentionRequests.state)}
          reason={arc.attentionRequests.reason}
        />
        <Stat
          id="R19"
          label="Off-lesson talk"
          value={
            arc.drift.value
              ? `${arc.drift.value.episodes} episode${arc.drift.value.episodes === 1 ? "" : "s"}`
              : "—"
          }
          sub={
            arc.drift.value
              ? `${msToClock(arc.drift.value.totalMs)} in total · admin talk stands in`
              : null
          }
          state={st(arc.drift.state)}
          reason={arc.drift.reason}
        />
      </div>

      {showEvidence && (
        <div className="mt-5 space-y-3 border-t border-border/60 pt-4">
          {evidence.map((g) => (
            <div key={g.title}>
              <p className="text-xs font-medium text-muted-foreground">{g.title}</p>
              <ul className="mt-1 space-y-0.5">
                {g.items.map((e) => (
                  <li key={e.idx} className="flex gap-2 text-xs leading-relaxed">
                    <button
                      type="button"
                      onClick={() => onSeek(e.atMs)}
                      className="shrink-0 font-mono text-muted-foreground tabular-nums hover:text-foreground"
                    >
                      {msToClock(e.atMs)}
                    </button>
                    <span className="text-muted-foreground" title={e.text}>
                      “{displayLine(e, true)}”
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
