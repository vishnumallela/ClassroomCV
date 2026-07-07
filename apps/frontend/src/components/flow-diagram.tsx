import { ArrowRight } from "lucide-react";

const STEPS = [
  { label: "Browser", detail: "Upload a recording" },
  { label: "API", detail: "Hono stores the file" },
  { label: "Queue", detail: "BullMQ on Redis" },
  { label: "Worker", detail: "Durable pipeline" },
  { label: "ML service", detail: "YOLO and SAM2" },
  { label: "TimescaleDB", detail: "Detections and analytics" },
  { label: "Dashboard", detail: "Video and charts" },
];

export function FlowDiagram() {
  return (
    <div className="overflow-x-auto pb-2">
      <div className="flex min-w-max items-center gap-2">
        {STEPS.map((step, i) => (
          <div key={step.label} className="flex items-center gap-2">
            <div className="w-36 rounded-lg border border-border bg-card p-3 text-center">
              <div className="text-sm font-medium">{step.label}</div>
              <div className="mt-1 text-xs text-muted-foreground">{step.detail}</div>
            </div>
            {i < STEPS.length - 1 && (
              <ArrowRight className="size-4 shrink-0 text-muted-foreground" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
