import { createFileRoute } from "@tanstack/react-router";
import { FlowDiagram } from "@/components/flow-diagram";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

export const Route = createFileRoute("/architecture")({ component: Architecture });

const STAGES = [
  {
    title: "Frame sampling",
    what: "ffmpeg reads the recording and samples about five frames per second.",
    why: "Full frame rate is redundant for behavior analysis, so a lower rate keeps the pipeline fast while still capturing movement.",
  },
  {
    title: "Person detection with pose",
    what: "YOLO11m-pose runs on each sampled frame and returns a bounding box plus seventeen body keypoints for every person.",
    why: "The keypoints reveal whether a person is standing or seated, which is the core signal for separating the teacher from students.",
  },
  {
    title: "Multi-object tracking",
    what: "BoT-SORT links detections across frames so each person keeps a stable track id while they remain visible.",
    why: "This turns thousands of per-frame boxes into a trajectory for each person.",
  },
  {
    title: "Re-identification",
    what: "When a person is occluded or walks out and back, the tracker issues a new id, so a merge step reunites the fragments using a torso color histogram, or spatial continuity when color is unavailable.",
    why: "Without it, a teacher who steps into the hallway and returns becomes two identities and her time is split in half. This is exactly how re-identification solves the leave and return problem.",
  },
  {
    title: "Teacher and student classification",
    what: "Each merged identity is scored on four behaviors: how much it stands, how far it moves across the room, how long it is present, and how often it stands at the board. A robust outlier rule promotes the clear leader to teacher, and short walk fragments are folded back into her identity.",
    why: "The teacher is the person who stands, moves, is present for most of the lesson, and works at the board, so behavior alone identifies her with no faces and no labels.",
  },
  {
    title: "Board zone detection",
    what: "SAM 2 proposes candidate regions and a geometric score selects the one shaped and placed like a board.",
    why: "Knowing where the board is lets the system measure time spent teaching at it, and the operator can correct the zone by hand.",
  },
  {
    title: "Event derivation",
    what: "The trajectories and zones become presence intervals, board intervals, entries and exits, and per second occupancy counts.",
    why: "These are the measured analytics the dashboard renders.",
  },
];

const STACK = [
  { layer: "Frontend", tech: "Vite, TanStack Router, TanStack Query, shadcn, Tailwind" },
  { layer: "API", tech: "Bun, Hono, oRPC, Drizzle" },
  { layer: "Queue", tech: "BullMQ on Redis, with a bull-board dashboard" },
  { layer: "ML service", tech: "FastAPI, Ultralytics YOLO11m-pose, SAM 2, ffmpeg" },
  { layer: "Database", tech: "TimescaleDB, a Postgres hypertable extension" },
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-4">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

function Architecture() {
  return (
    <div className="mx-auto max-w-4xl space-y-10">
      <header className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight">How the analytics work</h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          This tool turns a classroom recording into structured teaching analytics. An operator
          uploads a video, an automated pipeline measures the teacher and the students, and the
          dashboard shows the result. Nothing is labeled by hand.
        </p>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">5 frames per second</Badge>
          <Badge variant="secondary">YOLO11m-pose</Badge>
          <Badge variant="secondary">17 keypoints</Badge>
          <Badge variant="secondary">Per second occupancy</Badge>
        </div>
      </header>

      <Section title="The path a recording takes">
        <FlowDiagram />
        <p className="text-sm leading-relaxed text-muted-foreground">
          The browser uploads the video to the API, which stores the file and adds a job to the
          queue. A dedicated worker runs the analysis, calls the ML service for detection and
          tracking, and writes the results to the database. The dashboard then reads those results
          over a typed API.
        </p>
      </Section>

      <Section title="The detection pipeline">
        <p className="text-sm leading-relaxed text-muted-foreground">
          Inside the ML service, seven stages turn raw pixels into measured behavior. Each stage
          exists for a reason, described below.
        </p>
        <ol className="space-y-3">
          {STAGES.map((stage, i) => (
            <li key={stage.title}>
              <Card className="p-5">
                <div className="flex items-baseline gap-3">
                  <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/15 text-xs font-semibold text-primary">
                    {i + 1}
                  </span>
                  <h3 className="font-medium">{stage.title}</h3>
                </div>
                <p className="mt-2 text-sm leading-relaxed">{stage.what}</p>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  <span className="font-medium text-foreground">Why. </span>
                  {stage.why}
                </p>
              </Card>
            </li>
          ))}
        </ol>
      </Section>

      <Section title="How the data is stored">
        <Card className="p-5">
          <p className="text-sm leading-relaxed">
            The raw per-frame detections land in a TimescaleDB hypertable keyed on video time. A
            long lecture can produce hundreds of thousands of rows, and the hypertable partitions
            them by time so that reads for a single video stay fast. The small derived summaries,
            meaning the tracks, events, and the analytics for each video, live in ordinary
            relational tables. This keeps the heavy time series data separate from the compact
            results the dashboard reads on every load.
          </p>
        </Card>
      </Section>

      <Section title="Why the pipeline runs in a worker">
        <Card className="p-5">
          <p className="text-sm leading-relaxed">
            Video analysis is slow and GPU heavy, so it runs in a dedicated BullMQ worker that is
            separate from the web tier and can scale on its own. A fence token guards every write,
            so if an operator re-runs a video the older run stops before it can overwrite the newer
            result. The same worker model is what a live camera stream would use later, with frames
            arriving on the queue instead of a single uploaded file.
          </p>
        </Card>
      </Section>

      <Section title="Tech stack">
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <tbody className="divide-y divide-border">
                {STACK.map((row) => (
                  <tr key={row.layer}>
                    <td className="w-40 px-5 py-3 font-medium">{row.layer}</td>
                    <td className="px-5 py-3 text-muted-foreground">{row.tech}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </Section>
    </div>
  );
}
