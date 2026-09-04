import type { RouterOutputs } from "@classroom/api-contracts";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { msToClock } from "@/lib/format";
import { displayLine } from "@/lib/transcript";

type Arc = RouterOutputs["videos"]["get"]["arc"];
type Measure = {
  value: unknown;
  state: "observed" | "provisional" | "not_observed";
  reason: string | null;
  evidence: {
    idx: number;
    atMs: number;
    text: string;
    textEn?: string | null;
    language?: string | null;
  }[];
};

const CLOSURE_LABEL: Record<string, string> = {
  review: "Review",
  reflection: "Reflection",
  exit_question: "Exit question",
  summary: "Summary",
  none: "None — the lesson just stopped",
};

function StateBadge({ state }: { state: Measure["state"] }) {
  if (state === "observed") return null;
  return (
    <Badge variant={state === "provisional" ? "secondary" : "outline"} className="text-[0.6rem]">
      {state === "provisional" ? "provisional" : "not observed"}
    </Badge>
  );
}

function Row({
  id,
  label,
  value,
  measure,
  onSeek,
  clock,
}: {
  id: string;
  label: string;
  value: string;
  measure: Measure;
  onSeek: (ms: number) => void;
  clock: (ms: number) => string | null;
}) {
  const withheld = measure.state === "not_observed";
  return (
    <div className="py-2">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">
            <span className="mr-1.5 font-mono text-[0.65rem]">{id}</span>
            {label}
          </p>
          <p className={`text-sm font-medium ${withheld ? "text-muted-foreground" : ""}`}>
            {withheld ? "Not Observed" : value}
          </p>
        </div>
        <StateBadge state={measure.state} />
      </div>
      {measure.reason && (
        <p className="mt-0.5 text-[0.7rem] leading-relaxed text-muted-foreground">
          {measure.reason}
        </p>
      )}
      {measure.evidence.length > 0 && (
        <ul className="mt-1 space-y-0.5">
          {measure.evidence.map((e) => (
            <li key={e.idx} className="flex gap-2 text-[0.7rem] leading-relaxed">
              <button
                type="button"
                onClick={() => onSeek(e.atMs)}
                className="shrink-0 font-mono text-muted-foreground tabular-nums hover:text-foreground"
              >
                {msToClock(e.atMs)}
                {clock(e.atMs) ? ` · ${clock(e.atMs)}` : ""}
              </button>
              <span className="truncate text-muted-foreground" title={e.text}>
                “{displayLine(e, true)}”
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function signedMinutes(min: number, late: string, early: string): string {
  if (Math.abs(min) < 0.5) return "on the bell";
  const v = Math.abs(Math.round(min * 10) / 10);
  return `${v} min ${min > 0 ? late : early}`;
}

/**
 * Groups B and C — the lesson's start and end, and how it ended — with R18
 * and R19 beside them. Every row carries its state and the sentence it rests
 * on; a click on the time seeks the video to it, which is how a provisional
 * number gets checked.
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
  const clock = (ms: number): string | null =>
    recordingStartedAt
      ? new Date(new Date(recordingStartedAt).getTime() + ms).toLocaleTimeString("en-GB", {
          timeZone: timezone,
          hour: "2-digit",
          minute: "2-digit",
        })
      : null;
  const at = (ms: number | null) =>
    ms === null ? "—" : `${clock(ms) ?? msToClock(ms)}${clock(ms) ? ` (${msToClock(ms)} in)` : ""}`;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="p-5">
        <h2 className="font-display text-base font-semibold tracking-tight">The lesson</h2>
        <p className="text-xs text-muted-foreground">
          Start from her first task-setting sentence or her first writing or pointing at the board,
          whichever comes first — both together make it observed; end from the last teaching
          sentence or the last time she left the board.
        </p>
        <div className="mt-2 divide-y divide-border/60">
          <Row
            id="R7"
            label="Lesson start"
            value={
              at(arc.start.value) +
              (arc.start.corroborated
                ? " · voice and board agree"
                : arc.start.actionMs !== null && arc.start.voiceMs === null
                  ? " · from writing/pointing at the board"
                  : arc.start.voiceMs !== null && arc.start.actionMs === null
                    ? " · from her words alone"
                    : "")
            }
            measure={arc.start}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R8"
            label="Start delay"
            value={
              arc.startDelayMin.value !== null
                ? signedMinutes(arc.startDelayMin.value, "after the bell", "before the bell")
                : "—"
            }
            measure={{ ...arc.startDelayMin, reason: null, evidence: [] }}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R9"
            label="Lesson end"
            value={
              at(arc.end.value) + (arc.end.corroboratedByBoard ? " · board or exit confirms" : "")
            }
            measure={arc.end}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R10"
            label="Lesson duration"
            value={arc.durationMin.value !== null ? `${arc.durationMin.value} min` : "—"}
            measure={arc.durationMin}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R11"
            label="Did the lesson fit the period?"
            value={arc.fitsPeriod.value === null ? "—" : arc.fitsPeriod.value ? "Yes" : "No"}
            measure={arc.fitsPeriod}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R12"
            label="Overrun or underrun"
            value={
              arc.overrunMin.value !== null
                ? signedMinutes(arc.overrunMin.value, "past the end bell", "of the period unused")
                : "—"
            }
            measure={{ ...arc.overrunMin, evidence: [] }}
            onSeek={onSeek}
            clock={clock}
          />
        </div>
      </Card>

      <Card className="p-5">
        <h2 className="font-display text-base font-semibold tracking-tight">
          How the lesson ended
        </h2>
        <p className="text-xs text-muted-foreground">
          From the teacher's sentences, by phrase until the labelling pass exists.
        </p>
        <div className="mt-2 divide-y divide-border/60">
          <Row
            id="R13"
            label="Closure, and its type"
            value={
              arc.closure.value ? (CLOSURE_LABEL[arc.closure.value] ?? arc.closure.value) : "—"
            }
            measure={arc.closure}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R14"
            label="Continuation"
            value={
              arc.continuation.value === null
                ? "—"
                : arc.continuation.value
                  ? "Said the topic continues"
                  : "Not said"
            }
            measure={arc.continuation}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R15"
            label="Homework set"
            value={
              arc.homework.value === null
                ? "—"
                : arc.homework.value
                  ? `Yes${arc.homework.atMs !== null ? `, at ${at(arc.homework.atMs)}` : ""}`
                  : "No"
            }
            measure={arc.homework}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R18"
            label="Attention requests"
            value={
              arc.attentionRequests.value !== null
                ? `${arc.attentionRequests.value}` +
                  (arc.attentionRequests.perTenMinutes !== null
                    ? ` · ${arc.attentionRequests.perTenMinutes} per 10 min`
                    : "")
                : "—"
            }
            measure={arc.attentionRequests}
            onSeek={onSeek}
            clock={clock}
          />
          <Row
            id="R19"
            label="Off-lesson drift"
            value={
              arc.drift.value
                ? `${arc.drift.value.episodes} episode${arc.drift.value.episodes === 1 ? "" : "s"}, ${msToClock(arc.drift.value.totalMs)}`
                : "—"
            }
            measure={arc.drift}
            onSeek={onSeek}
            clock={clock}
          />
        </div>
      </Card>
    </div>
  );
}
