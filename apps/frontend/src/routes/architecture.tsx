import { createFileRoute } from "@tanstack/react-router";
import {
  Ban,
  Boxes,
  Camera,
  Fingerprint,
  Footprints,
  Layers,
  MapPin,
  Ruler,
  ScanFace,
  Scale,
  Sparkles,
  Timer,
  UserSearch,
} from "lucide-react";
import type { CSSProperties } from "react";
import { FlowDiagram } from "@/components/flow-diagram";
import { LuminaryMark } from "@/components/luminary-logo";
import { Card } from "@/components/ui/card";

export const Route = createFileRoute("/architecture")({ component: Architecture });

// --------------------------------------------------------------------------- //
// The models. Each is explained plainly first, then named and specified, so a
// curious head teacher and a curious engineer both leave satisfied.
// --------------------------------------------------------------------------- //
const MODELS = [
  {
    icon: Camera,
    name: "The Spotter",
    grownup: "YOLO26-pose",
    plain:
      "A single neural network that looks at one still frame and, in one pass, draws a tight box around every person and places 17 labelled dots on each body: eyes, nose, shoulders, elbows, wrists, hips, knees, ankles.",
    does: "It answers two questions per frame at once: where is each person, and what shape is their body making. The 17 dots (a skeleton) are what later let the system tell a standing adult apart from a seated child without ever looking at a face.",
    detail:
      "Ultralytics YOLO26-pose, the latest generation (NMS-free, up to +7.2 keypoint AP over YOLO11). It runs at 1280 to 1536 px so a 40 px back-row head survives the downscale, in half precision. It is device-aware: the large yolo26x variant on a production GPU (exported to TensorRT for roughly a 5x fp16 speedup), a lighter yolo26m for laptop development. A confidence floor and a 100-detection cap per frame keep the far rows in without flooding the tracker.",
  },
  {
    icon: Boxes,
    name: "The Tracker",
    grownup: "BoT-SORT",
    plain:
      "A short-term memory. In every new frame it compares the fresh boxes to the ones from the previous frame and decides which box is the same person continuing to move, then keeps a numbered sticker on them.",
    does: "It turns thousands of disconnected boxes into one continuous path per person. Without it there is no notion of a person over time, only a blizzard of unrelated rectangles.",
    detail:
      "A Kalman motion filter predicts where each person moves next, and boxes are matched to predictions by overlap (IoU). Because the camera is bolted to the wall, camera-motion compensation is switched off, and a lost track is held for several seconds so a person who ducks behind a desk is re-attached rather than renumbered.",
  },
  {
    icon: UserSearch,
    name: "The Reuniter",
    grownup: "CLIP embedding + seat anchors",
    plain:
      "A memory for people who disappear and come back. When someone is hidden for a while or walks out and returns, the Tracker gives them a brand new number. The Reuniter decides which of those pieces are actually the same person and stitches them back into one identity.",
    does: "It stops a teacher who steps into the corridor and returns from being counted as two people (which would halve her measured time), and stops a student who leans out of view from being double counted.",
    detail:
      "Every fragment pair is scored on appearance (a CLIP image embedding plus an HSV torso-colour histogram) and, decisively for a room in identical uniforms where appearance is nearly useless, on spatial continuity and a seat anchor: two stationary fragments sitting at different desks are refused outright, no matter how similar their shirts, because a seat is the one thing a uniform cannot fake.",
  },
  {
    icon: ScanFace,
    name: "The Zone Finder",
    grownup: "YOLO-World + SAM 2",
    plain:
      "The part that finds the board and the door. One model is told, in plain English, to find a chalkboard, and points roughly at it. A second model traces its exact outline. A small rulebook then checks the shape is really a board.",
    does: "It gives the room a front (the board) and its exits (the doors), so standing at the front becomes measured teaching time and a disappearance near a door becomes a confirmed exit.",
    detail:
      "YOLO-World does open-vocabulary detection from a text prompt, SAM 2 segments each candidate into a precise mask, and a geometric score (wide, high on the wall, rectangular, flat-coloured for a board; tall and narrow for a door) picks the winner and rejects false positives. An operator can redraw either zone by hand at any time.",
  },
];

// --------------------------------------------------------------------------- //
// The four behavioural signals that separate the teacher from the students.
// Values and weights are the real ones from services/ml-service/app/roles.py.
// --------------------------------------------------------------------------- //
const SIGNALS = [
  {
    icon: Footprints,
    label: "Stands up",
    weight: "0.30",
    measures: "The fraction of the lesson a person is standing.",
    how: "A person counts as standing when their box is taller than it is wide past a ratio of 1.6, or when the skeleton shows the hips clearly above the knees. Because a single frame can flicker, the flag is smoothed with a 5-sample majority vote before it is counted.",
    teacher: "0.74",
    student: "0.05",
  },
  {
    icon: MapPin,
    label: "Roams the room",
    weight: "0.25",
    measures: "How far across the room a person travels over the whole lesson.",
    how: "Measured as the spatial extent of the path (the width or height of the region the person's centre visits), not the wandering distance, so a seated child fidgeting in place scores near zero. It is normalised so covering 40 percent of the frame already saturates the signal.",
    teacher: "0.98",
    student: "0.04",
  },
  {
    icon: Timer,
    label: "Is present throughout",
    weight: "0.25",
    measures: "How much of the lesson the person is on camera at all.",
    how: "The span from a person's first sighting to their last, divided by the lesson length. A teacher is there start to finish; most students arrive and leave in a narrower window, and passers-by barely register.",
    teacher: "0.98",
    student: "0.90",
  },
  {
    icon: Scale,
    label: "Works at the board",
    weight: "0.20",
    measures: "The fraction of the lesson spent standing at the front of the room.",
    how: "A sample counts only when the person is standing, their centre sits inside the board's horizontal span, and the bottom of their box reaches the floor line of the board (so a seated child in the middle of the room cannot score). If no board zone exists, this signal is dropped and the other three are re-weighted.",
    teacher: "0.19",
    student: "0.00",
  },
];

// --------------------------------------------------------------------------- //
// The pipeline stages, each explained in real depth.
// --------------------------------------------------------------------------- //
const STAGES = [
  {
    title: "Sample the frames",
    what: "ffmpeg reads the recording and pulls out roughly five frames every second instead of the full thirty.",
    why: "Teaching behaviour changes over seconds, not milliseconds, so five frames a second captures every entrance, every walk to the board, and every posture change while doing one sixth of the work. That ratio is exactly what makes analysing an eight hour day across many rooms affordable rather than ruinous.",
  },
  {
    title: "Find every person and their pose",
    what: "The Spotter runs on each sampled frame and returns, for every person, a bounding box plus the 17 body keypoints.",
    why: "This is the only stage that looks at raw pixels. Everything downstream reasons about boxes and skeletons, never faces. Running at high resolution here is the single biggest quality lever in the whole system, because a back-row student who is only 40 pixels tall is either seen now or lost forever.",
  },
  {
    title: "Follow each person through time",
    what: "The Tracker links the per-frame boxes into one continuous trajectory per person and gives each a stable id.",
    why: "A metric like time at the board only means something for a person you can follow. This stage converts tens of thousands of independent detections into a few dozen moving paths, which is the object every later stage actually works on.",
  },
  {
    title: "Reunite the broken pieces",
    what: "The Reuniter merges the fragments that occlusion and exits inevitably create back into stable identities, using position and a seat anchor as much as appearance.",
    why: "Trackers break a person into new ids the moment they are hidden or leave the frame. Left alone, that fragmentation would split one teacher into a dozen identities and inflate the student count. This stage is where a room full of identical uniforms is handled honestly: it leans on where people are, not what colour they wear.",
  },
  {
    title: "Decide who is the teacher",
    what: "Each stable identity is scored on four behaviours, and the one clear behavioural outlier is labelled the teacher. Everyone else is a student, and if nobody stands out, nobody is labelled.",
    why: "The teacher never wears a badge the system can read, so identity is inferred from conduct: standing, roaming, being present throughout, and working at the board. The full rule is spelled out in the next section, because it is the heart of the product and the place a school leader most deserves to see the workings.",
  },
  {
    title: "Turn movement into measured events",
    what: "The teacher's trajectory is crossed with the zones to produce the final numbers: presence intervals, entries and exits at the door, and time at the board, opened and closed with hysteresis so a single occluded frame never fabricates an event.",
    why: "This is where paths become the figures on the dashboard. Every rule here exists to stop a tracking hiccup from becoming a false story, which is the difference between a number a principal can act on and one they cannot.",
  },
];

const NOT_DOING = [
  "Recognise faces or identify any student by name",
  "Score a child's attention, engagement, or emotion (video cannot honestly measure this)",
  "Produce a per-student attendance register",
  "Claim distances in feet without camera calibration",
  "Keep student appearance fingerprints after the video is processed",
];

const STACK = [
  { layer: "Frontend", tech: "Vite, TanStack Router and Query, shadcn, Tailwind" },
  { layer: "API", tech: "Bun, Hono, oRPC, Drizzle" },
  { layer: "Queue", tech: "BullMQ on Redis, with a live job dashboard" },
  { layer: "ML service", tech: "FastAPI, Ultralytics YOLO26-pose, SAM 2, CLIP, ffmpeg" },
  { layer: "Database", tech: "TimescaleDB, a time-series Postgres for the detection firehose" },
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="reveal space-y-5">
      <h2 className="font-display text-2xl font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.82em] text-foreground">
      {children}
    </code>
  );
}

function Architecture() {
  return (
    <div className="mx-auto max-w-4xl space-y-16">
      {/* Hero */}
      <header className="reveal space-y-5">
        <LuminaryMark className="size-11" bloom />
        <div className="space-y-3">
          <h1 className="font-display text-[2.5rem] font-semibold leading-[1.05] tracking-tight">
            How Luminary reads a lesson
          </h1>
          <p className="max-w-2xl text-base leading-relaxed text-muted-foreground">
            You give it a classroom recording. It hands back a clear picture of the teaching: how
            present the teacher was, how long she spent at the board, how the class filled and
            settled, and how she moved through the room. Every figure is measured automatically,
            nobody labels anything by hand, and no face is ever recognised.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          {["5 frames / second", "17 body points", "No facial recognition", "Aggregate only"].map(
            (b) => (
              <span
                key={b}
                className="rounded-full border border-border bg-card px-3 py-1 font-mono text-muted-foreground"
              >
                {b}
              </span>
            ),
          )}
        </div>
      </header>

      {/* Big picture */}
      <Section title="The idea, in one paragraph">
        <Card className="p-6">
          <p className="text-base leading-relaxed">
            Imagine an assistant who can watch a whole lesson without ever getting bored or
            distracted. Five times a second they glance at the room and note where every person is
            and whether they are standing or sitting. They never learn a single name; they only
            follow shapes moving around. At the end they add it up. The person who stood, walked the
            room, stayed the whole time, and worked at the board was the teacher. Everyone seated
            was a student. Luminary is that assistant, built from a small line of specialised AI
            models, each doing one job and handing off to the next.
          </p>
        </Card>
      </Section>

      {/* Journey */}
      <Section title="The journey of a recording">
        <FlowDiagram />
        <p className="text-sm leading-relaxed text-muted-foreground">
          The browser uploads the video to the API, which stores it and adds a job to a queue. A
          dedicated worker runs the analysis, calls the vision service for detection and tracking,
          and writes the results to the database. The dashboard then reads those results over a
          typed API, so the heavy work never blocks the page.
        </p>
      </Section>

      {/* Models */}
      <Section title="The four specialists doing the work">
        <p className="-mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Each specialist does one job well, then passes its output to the next. Here is each one
          explained plainly, then named and specified.
        </p>
        <div className="stagger grid gap-4 sm:grid-cols-2">
          {MODELS.map((m, i) => (
            <Card
              key={m.name}
              className="flex flex-col gap-3 p-5 transition-colors hover:border-primary/40"
              style={{ "--i": i } as CSSProperties}
            >
              <div className="flex items-center justify-between">
                <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <m.icon className="size-5" />
                </span>
                <code className="rounded-md bg-muted px-2 py-1 font-mono text-[0.7rem] text-muted-foreground">
                  {m.grownup}
                </code>
              </div>
              <h3 className="font-display text-lg font-semibold tracking-tight">{m.name}</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">{m.plain}</p>
              <p className="text-sm leading-relaxed">
                <span className="font-medium">What it does here. </span>
                {m.does}
              </p>
              <p className="mt-auto flex gap-2 border-t border-border pt-3 text-xs leading-relaxed text-muted-foreground">
                <Sparkles className="mt-px size-3.5 shrink-0 text-light" />
                {m.detail}
              </p>
            </Card>
          ))}
        </div>
      </Section>

      {/* Pipeline stages */}
      <Section title="The six stages, from pixels to numbers">
        <ol className="stagger space-y-3">
          {STAGES.map((stage, i) => (
            <li key={stage.title} style={{ "--i": i } as CSSProperties}>
              <Card className="flex gap-4 p-5">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/12 font-mono text-sm font-semibold text-primary">
                  {i + 1}
                </span>
                <div>
                  <h3 className="font-medium">{stage.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed">{stage.what}</p>
                  <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                    <span className="font-medium text-foreground">Why it matters. </span>
                    {stage.why}
                  </p>
                </div>
              </Card>
            </li>
          ))}
        </ol>
      </Section>

      {/* The teacher heuristic: the star technical section */}
      <Section title="How Luminary tells the teacher from the students">
        <p className="-mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
          This is the heart of the system, so here it is in full. There is no face recognition and
          no name tag. A teacher is identified purely by how she behaves, because in a classroom the
          teacher does four things almost nobody else does at once.
        </p>

        <div className="stagger grid gap-4 sm:grid-cols-2">
          {SIGNALS.map((s, i) => (
            <Card
              key={s.label}
              className="flex flex-col gap-3 p-5"
              style={{ "--i": i } as CSSProperties}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5">
                  <span className="flex size-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <s.icon className="size-[1.05rem]" />
                  </span>
                  <h3 className="font-medium">{s.label}</h3>
                </div>
                <span
                  className="rounded-md bg-primary/10 px-2 py-1 font-mono text-xs font-medium text-primary"
                  title="Weight this signal carries in the combined score"
                >
                  weight {s.weight}
                </span>
              </div>
              <p className="text-sm leading-relaxed">{s.measures}</p>
              <p className="text-sm leading-relaxed text-muted-foreground">{s.how}</p>
              {/* teacher vs student example bar */}
              <div className="mt-auto space-y-2 border-t border-border pt-3">
                <SignalBar label="Typical teacher" value={Number(s.teacher)} tone="primary" />
                <SignalBar label="Typical student" value={Number(s.student)} tone="muted" />
              </div>
            </Card>
          ))}
        </div>

        {/* The decision rule */}
        <Card className="space-y-4 p-6">
          <div className="flex items-center gap-2">
            <Scale className="size-4 text-primary" />
            <h3 className="font-medium">The decision rule</h3>
          </div>
          <p className="text-sm leading-relaxed">
            The four signals are combined into one score per person, each multiplied by its weight
            and added together:
          </p>
          <div className="overflow-x-auto rounded-lg bg-muted/60 p-4 font-mono text-xs leading-relaxed text-foreground">
            <div>score = 0.30 &times; stands</div>
            <div className="pl-[3.6rem]">+ 0.25 &times; roams</div>
            <div className="pl-[3.6rem]">+ 0.25 &times; present</div>
            <div className="pl-[3.6rem]">+ 0.20 &times; at_board</div>
            <div className="mt-3 text-muted-foreground">
              # teacher chosen only when the top score is a genuine outlier:
            </div>
            <div>
              teacher = best if <span className="text-primary">best &ge; 0.50</span> and{" "}
              <span className="text-primary">
                (best - runner_up) &ge; max(0.08, 0.15 &times; best)
              </span>
            </div>
            <div className="text-muted-foreground">else: nobody is labelled the teacher</div>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            The second condition is the important one. It is not enough to have the highest score;
            the top person has to{" "}
            <span className="font-medium text-foreground">
              lead the runner-up by a clear margin
            </span>
            . On the demo lesson the teacher scores about <Mono>0.71</Mono> while the most
            teacher-like student chain reaches only about <Mono>0.59</Mono>, a lead well past the
            threshold. When two adults share a room, or when a lively student presents at the front,
            the margin collapses and Luminary declines to guess: it labels everyone{" "}
            <Mono>unknown</Mono> and the teacher metrics read as unavailable rather than wrong. The
            size of that winning margin becomes the{" "}
            <span className="font-medium text-foreground">confidence</span> shown next to the
            teacher figure on the dashboard, as <Mono>0.5 + margin</Mono>.
          </p>
        </Card>

        {/* The gates */}
        <Card className="space-y-3 p-6">
          <div className="flex items-center gap-2">
            <Fingerprint className="size-4 text-primary" />
            <h3 className="font-medium">Who is not even allowed to be the teacher</h3>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            Before any scoring, obvious non-candidates are filtered out so a brief glitch can never
            be crowned. Three gates apply. Each of these may still be labelled a student; they
            simply cannot claim or block the teacher slot.
          </p>
          <ul className="grid gap-2.5 sm:grid-cols-3">
            {[
              {
                t: "Too brief",
                d: "Seen for less than about 60 seconds. A passer-by or a one-off fragment is not a teacher.",
                m: "span < 60 s",
              },
              {
                t: "Stuck to an edge",
                d: "Average position hugging a frame edge, where clipped boxes always look tall and fake-standing.",
                m: "edge < 0.03",
              },
              {
                t: "Too small",
                d: "A box far smaller than the median person, so a distant back-row head cannot outscore the adult.",
                m: "area < 0.3 x median",
              },
            ].map((g) => (
              <li key={g.t} className="rounded-lg border border-border bg-background/50 p-3">
                <div className="text-sm font-medium">{g.t}</div>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{g.d}</p>
                <code className="mt-2 inline-block font-mono text-[0.7rem] text-primary">
                  {g.m}
                </code>
              </li>
            ))}
          </ul>
          <p className="text-sm leading-relaxed text-muted-foreground">
            One last safeguard runs after the teacher is chosen. Because she walks in and out of
            view, the tracker often hands her return a fresh id that the Reuniter cannot bridge
            across a long gap. A dedicated step re-claims those stray fragments that fall inside her
            absence windows near the door, the board, or her own path, so she stays a single
            continuous identity across the whole lesson rather than being downgraded to a student
            halfway through.
          </p>
        </Card>
      </Section>

      {/* Trust */}
      <Section title="How Luminary shows its confidence">
        <Card className="p-6">
          <div className="flex items-start gap-3">
            <Fingerprint className="mt-0.5 size-5 shrink-0 text-primary" />
            <p className="text-base leading-relaxed">
              A number is only useful if you know how much to trust it. Every analysed lesson
              carries a confidence report built from three honest signals: how much of the lesson
              the camera actually covered, how cleanly the tracker followed people without
              fragmenting them, and a re-identification-independent head count (the most people
              visible in any single frame, which can never double-count anyone) as a cross-check on
              the identity-based count. When those disagree, the dashboard says so out loud rather
              than showing a falsely precise figure.
            </p>
          </div>
        </Card>
      </Section>

      {/* Storage */}
      <Section title="Where the data lives, and how it survives 80 classrooms">
        <Card className="p-6">
          <p className="text-base leading-relaxed">
            A single lesson produces hundreds of thousands of detection rows, and a live camera
            would produce them forever. So storage follows the standard three-tier split used across
            professional video analytics, on a TimescaleDB hypertable that ages and compresses data
            automatically.
          </p>
          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            {[
              {
                icon: Layers,
                title: "Hot tier",
                body: "Raw per-frame detections. Compressed after a day, dropped after a week, enough to audit and re-derive recent footage.",
              },
              {
                icon: Ruler,
                title: "Overlay tier",
                body: "Simplified track paths and sparse keyframe boxes. A few percent of the raw size, kept forever, so playback overlays survive after the raw rows age out.",
              },
              {
                icon: Fingerprint,
                title: "Aggregate tier",
                body: "Events, track summaries, and per-lesson analytics. Kept forever at negligible size. This is everything the dashboard shows.",
              },
            ].map((t) => (
              <div key={t.title} className="rounded-xl border border-border bg-background/50 p-4">
                <t.icon className="size-5 text-primary" />
                <div className="mt-2 text-sm font-medium">{t.title}</div>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{t.body}</p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
            Because the two permanent tiers hold everything you can see, retention never erases a
            number anyone relies on, and the raw firehose stays bounded no matter how many hours of
            how many rooms flow through it.
          </p>
        </Card>
      </Section>

      {/* Boundaries */}
      <Section title="What Luminary will not do">
        <Card className="border-destructive/20 bg-destructive/[0.03] p-6">
          <p className="text-sm leading-relaxed text-muted-foreground">
            The most important design choices are the ones we refused. Luminary deliberately does
            not:
          </p>
          <ul className="mt-4 grid gap-2.5 sm:grid-cols-2">
            {NOT_DOING.map((item) => (
              <li key={item} className="flex items-start gap-2.5 text-sm">
                <Ban className="mt-0.5 size-4 shrink-0 text-destructive/70" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
          <p className="mt-4 border-t border-border pt-3 text-sm leading-relaxed">
            These are not limitations to apologise for. They are the reason a school can adopt
            Luminary without a privacy fight. Insights are aggregate, teacher-facing, and about the{" "}
            <span className="font-medium">craft of teaching</span>, never about surveilling
            children.
          </p>
        </Card>
      </Section>

      {/* Stack */}
      <Section title="The stack">
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <tbody className="divide-y divide-border">
                {STACK.map((row) => (
                  <tr key={row.layer}>
                    <td className="w-36 px-5 py-3.5 font-medium">{row.layer}</td>
                    <td className="px-5 py-3.5 font-mono text-[0.8rem] text-muted-foreground">
                      {row.tech}
                    </td>
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

function SignalBar({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "primary" | "muted";
}) {
  const pct = Math.max(2, Math.round(value * 100));
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-24 shrink-0 text-xs text-muted-foreground">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
        <div
          className={
            tone === "primary"
              ? "h-full rounded-full bg-primary"
              : "h-full rounded-full bg-muted-foreground/40"
          }
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-9 shrink-0 text-right font-mono text-xs tabular-nums text-foreground">
        {value.toFixed(2)}
      </span>
    </div>
  );
}
