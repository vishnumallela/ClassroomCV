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
    what: "When a person is occluded or walks out and back, the tracker issues a new id, so a merge step reunites fragments into one identity. Every candidate pair is scored on four terms at once: appearance (torso color histogram, 35 percent), spatial continuity between where one fragment ended and the next began (25 percent), body size similarity (20 percent), and how close in time they are (20 percent). Two extra classroom rules apply: a fast moving person like the teacher gets a relaxed spatial requirement so her long walks still reconnect, and two stationary fragments anchored at different desks are refused outright, no matter how alike their shirts look.",
    why: "Students wear uniforms, so color alone cannot tell them apart; two students in identical red shirts would otherwise merge into one person across the room. Position is the evidence uniforms cannot fake: a seat is an identity anchor. Without re-identification, a teacher who steps into the hallway and returns becomes two identities and her time is split in half.",
  },
  {
    title: "Teacher and student classification",
    what: "Each merged identity is scored on four behaviors with fixed weights: how much of the time it stands (30 percent), how far it ranges across the room (25 percent), how long it is present in the video (25 percent), and how often it stands at the board (20 percent). Before scoring, obvious non-candidates are gated out: short lived fragments, frame edge slivers, and boxes far smaller than the median person. The standing signal itself is smoothed with a one second majority vote so single frame pose flickers do not count. The best candidate becomes the teacher only when its score clears an absolute floor and leads the runner up by a clear outlier margin; otherwise everyone stays unlabeled rather than guessing.",
    why: "The teacher is the person who stands, patrols, is present for most of the lesson, and works at the board. That behavioral signature identifies her with no faces and no manual labels, and the outlier rule means the system degrades gracefully instead of promoting a random student when nothing truly looks like teaching.",
  },
  {
    title: "Board and door zone detection",
    what: "YOLO World proposes candidate regions from text prompts like chalkboard or classroom door, SAM 2 traces each candidate into a precise outline, and a geometric score picks the winner: boards must be wide, high on the wall, rectangular, and flat colored; doors must be tall, narrow, and reach toward the floor. When the open vocabulary model finds nothing, a grid of SAM 2 probes with the same geometric scoring takes over.",
    why: "The board zone is what turns standing at the front into measured teaching time, and the door zone is what turns a disappearance into a confirmed exit. The operator can always redraw either zone by hand.",
  },
  {
    title: "Event derivation",
    what: "Trajectories and zones become the final analytics. Presence intervals split wherever the teacher vanishes for five seconds or more. Board time uses hysteresis, two seconds sustained to open and three to close, and now tolerates single frame occlusion flickers. Entries and exits count only presence edges where the teacher was near a door within four seconds, so a mid room occlusion is not mistaken for leaving. Occupancy counts distinct students per five second bucket.",
    why: "These are the measured numbers the dashboard renders, and each rule exists to keep a tracking hiccup from becoming a false event.",
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
            The raw per-frame detections land in a TimescaleDB hypertable. A long lecture produces
            hundreds of thousands of rows, and a live camera would produce them forever, so the
            storage follows the standard video analytics split into three tiers. Raw detections
            are the hot tier: a ring buffer that gets compressed after a day and dropped after a
            week, enough to audit and re-derive recent footage. Simplified track paths and sparse
            keyframe boxes are the permanent overlay tier, a few percent of the raw size, which
            keeps playback overlays working after the raw rows age out. Events, track summaries,
            and per video analytics are the permanent aggregate tier, kept forever at negligible
            size. Everything the dashboard shows lives in the two permanent tiers, so retention
            never erases a number anyone can see.
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
