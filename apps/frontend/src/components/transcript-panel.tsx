import type { RouterOutputs } from "@classroom/api-contracts";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { msToClock } from "@/lib/format";

type Transcript = RouterOutputs["videos"]["get"]["transcript"];

const LANGUAGE_LABEL: Record<string, string> = { hi: "hi", en: "en", mixed: "hi+en" };

type Filter = "all" | "teacher" | "others";

/**
 * The transcript as sentences, the evidence behind every voice number.
 *
 * A row's time seeks the video, so a number on the card can be checked
 * against what was actually said and who was on screen when. "Teacher" is
 * the voice the card resolved; the raw diarizer label stays visible so a
 * wrong resolution can be seen rather than trusted.
 */
export function TranscriptPanel({
  transcript,
  onSeek,
}: {
  transcript: Transcript;
  onSeek: (ms: number) => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const rows = transcript.filter((u) =>
    filter === "all" ? true : filter === "teacher" ? u.isTeacher === true : u.isTeacher !== true,
  );

  if (transcript.length === 0) return null;

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-1">
          <h2 className="font-display text-base font-semibold tracking-tight">Transcript</h2>
          <p className="text-xs text-muted-foreground">
            {transcript.length} sentences. Click a time to jump the video there.
          </p>
        </div>
        <div className="flex gap-1">
          {(["all", "teacher", "others"] as Filter[]).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={
                "rounded-md border px-2 py-0.5 text-xs capitalize transition-colors " +
                (f === filter
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-input bg-background text-muted-foreground hover:text-foreground")
              }
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <ol className="mt-3 max-h-[28rem] divide-y divide-border/60 overflow-y-auto text-sm">
        {rows.map((u) => (
          <li key={u.idx} className="flex gap-3 py-1.5">
            <button
              type="button"
              onClick={() => onSeek(u.startMs)}
              className="w-12 shrink-0 text-left font-mono text-xs text-muted-foreground tabular-nums hover:text-foreground"
            >
              {msToClock(u.startMs)}
            </button>
            <span className="w-16 shrink-0">
              <Badge
                variant={u.isTeacher ? "default" : "secondary"}
                className="text-[0.6rem]"
                title={`Speaker ${u.speaker}`}
              >
                {u.isTeacher ? "Teacher" : `Other ${u.speaker}`}
              </Badge>
            </span>
            <span className="min-w-0 flex-1 leading-snug">
              {u.text}
              {u.language && (
                <span className="ml-1.5 align-middle text-[0.6rem] uppercase text-muted-foreground/70">
                  {LANGUAGE_LABEL[u.language] ?? u.language}
                </span>
              )}
            </span>
          </li>
        ))}
      </ol>
    </Card>
  );
}
