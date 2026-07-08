import { Boxes, Clapperboard, Database, LayoutDashboard, ScanLine, Workflow } from "lucide-react";
import type { CSSProperties } from "react";

const STEPS = [
  { label: "Recording", detail: "An operator uploads a video", icon: Clapperboard },
  { label: "Intake", detail: "The API stores it and queues a job", icon: Workflow },
  { label: "Worker", detail: "A durable worker runs the pipeline", icon: Boxes },
  { label: "Vision", detail: "Models find & follow every person", icon: ScanLine },
  { label: "Store", detail: "Detections + insights land in the DB", icon: Database },
  { label: "Dashboard", detail: "You read the lesson", icon: LayoutDashboard },
];

export function FlowDiagram() {
  return (
    <div className="overflow-x-auto pb-1">
      <ol className="stagger flex min-w-max items-stretch gap-1.5">
        {STEPS.map((step, i) => (
          <li
            key={step.label}
            className="flex items-center gap-1.5"
            style={{ "--i": i } as CSSProperties}
          >
            <div className="flex w-36 flex-col gap-2 rounded-xl border border-border bg-card p-3.5">
              <span className="flex size-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <step.icon className="size-[1.05rem]" />
              </span>
              <div>
                <div className="text-sm font-medium">{step.label}</div>
                <div className="mt-0.5 text-xs leading-snug text-muted-foreground">
                  {step.detail}
                </div>
              </div>
            </div>
            {i < STEPS.length - 1 && (
              <span
                aria-hidden
                className="h-px w-4 shrink-0 bg-gradient-to-r from-border to-primary/40"
              />
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
