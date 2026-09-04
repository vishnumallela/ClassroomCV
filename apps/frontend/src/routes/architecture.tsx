import { createFileRoute } from "@tanstack/react-router";
import { ArchitectureDiagram } from "@/components/architecture-diagram";
import { Card } from "@/components/ui/card";
import { KPIS, SECTIONS, VOICE_EXTRAS, type Step } from "@/lib/how-it-works";

export const Route = createFileRoute("/architecture")({ component: HowItWorks });

/**
 * The system end to end, for a developer who has never opened the code: a
 * diagram of the whole data flow, then every stage in the order it runs —
 * what goes in, what happens, what comes out, where it goes next, who owns
 * it, and the constants that decide it. Content lives in lib/how-it-works.ts.
 */

const FLOW = [
  "Video / audio in",
  "Ingestion",
  "Preprocessing",
  "AI model",
  "Detection",
  "Tracking",
  "Event logic",
  "Analytics",
  "KPI generation",
  "Database",
  "API",
  "UI",
];

function StepCard({ step, i }: { step: Step; i: number }) {
  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-display text-base font-semibold tracking-tight">
          <span className="mr-2 font-mono text-xs text-primary">{i + 1}</span>
          {step.name}
        </h3>
        <span className="font-mono text-[0.7rem] text-muted-foreground">{step.owner}</span>
      </div>
      <dl className="mt-3 grid gap-x-6 gap-y-2 text-sm leading-relaxed sm:grid-cols-[6.5rem_1fr]">
        <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">In</dt>
        <dd>{step.in}</dd>
        <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Processing
        </dt>
        <dd>{step.processing}</dd>
        <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Out</dt>
        <dd>{step.out}</dd>
        <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Next</dt>
        <dd>{step.next}</dd>
      </dl>
      {step.config && step.config.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {step.config.map((c) => (
            <span
              key={c}
              className="rounded-md border border-border bg-muted/40 px-2 py-0.5 font-mono text-[0.68rem] text-muted-foreground"
            >
              {c}
            </span>
          ))}
        </div>
      )}
    </Card>
  );
}

function KpiTable({ rows, title }: { rows: typeof KPIS; title: string }) {
  return (
    <Card className="overflow-hidden">
      <div className="border-b border-border bg-muted/40 px-3 py-2 text-xs font-medium text-muted-foreground">
        {title}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border font-mono text-[0.68rem] uppercase tracking-wider text-muted-foreground">
            <tr>
              <th className="px-3 py-2 font-medium">Id</th>
              <th className="px-3 py-2 font-medium">Measurement</th>
              <th className="px-3 py-2 font-medium">Source data</th>
              <th className="px-3 py-2 font-medium">Calculation</th>
              <th className="px-3 py-2 font-medium">State</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {rows.map((r) => (
              <tr key={`${r.id}-${r.name}`} className="align-top">
                <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-primary">
                  {r.id}
                </td>
                <td className="whitespace-nowrap px-3 py-2 font-medium">{r.name}</td>
                <td className="px-3 py-2 font-mono text-[0.7rem] leading-relaxed text-muted-foreground">
                  {r.source}
                </td>
                <td className="px-3 py-2 text-xs leading-relaxed">{r.calc}</td>
                <td className="whitespace-nowrap px-3 py-2 text-xs text-muted-foreground">
                  {r.state}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function HowItWorks() {
  return (
    <div className="mx-auto max-w-5xl space-y-12">
      <header className="space-y-4">
        <div className="font-mono text-xs uppercase tracking-widest text-primary">
          System architecture · end to end
        </div>
        <h1 className="font-display text-3xl font-semibold tracking-tight">How it works</h1>
        <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">
          From the moment a recording enters to the numbers on the lesson page, in the order it
          runs. Every stage says what goes in, what happens, what comes out, where it goes next,
          which service owns it, and the constants that decide it — the values named here are the
          ones the code runs. Two pipelines run in parallel: the video half on a rented GPU, the
          audio half on the API host; they meet in the database, and every number is derived from
          stored rows at read time, so a changed rule is a re-read, never a re-run.
        </p>
        <div className="flex flex-wrap items-center gap-1.5 text-[0.7rem]">
          {FLOW.map((f, i) => (
            <span key={f} className="flex items-center gap-1.5">
              <span className="rounded-md border border-border bg-muted/40 px-2 py-0.5 font-mono">
                {f}
              </span>
              {i < FLOW.length - 1 && <span className="text-muted-foreground">→</span>}
            </span>
          ))}
        </div>
        <nav className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          <a href="#diagram" className="text-muted-foreground underline-offset-2 hover:underline">
            Diagram
          </a>
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className="text-muted-foreground underline-offset-2 hover:underline"
            >
              {s.n} · {s.title}
            </a>
          ))}
        </nav>
      </header>

      <section id="diagram" className="space-y-3">
        <h2 className="font-display text-xl font-semibold tracking-tight">
          Architecture and data flow
        </h2>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Read left to right within a lane and top to bottom between lanes. The video lane wraps
          once: frame → detection → appearance → zones → tracking → attribution, then events →
          per-person analytics → quality → persist. The audio lane runs beside it and both land in
          the same database, where the read-time lane turns rows into the period's numbers.
        </p>
        <ArchitectureDiagram />
      </section>

      {SECTIONS.map((s) => (
        <section key={s.id} id={s.id} className="space-y-4">
          <div className="space-y-1">
            <h2 className="font-display text-xl font-semibold tracking-tight">
              <span className="mr-2 font-mono text-sm text-primary">{s.n}</span>
              {s.title}
            </h2>
            <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">{s.intro}</p>
          </div>
          {s.id === "kpis" ? (
            <div className="space-y-4">
              <KpiTable rows={KPIS} title="The 23 measurements (docs/teacher-measurements.md)" />
              <KpiTable rows={VOICE_EXTRAS} title="Voice figures shown beside them" />
              <p className="text-xs leading-relaxed text-muted-foreground">
                Chain of custody: boxes (RF-DETR) → people (tracker + attribution) → intervals and
                events (heuristics) → stored rows → numbers (API dto.ts, lib/voice.ts,
                lib/lesson-arc.ts, lib/trust.ts, lib/register.ts) → the page. A number that rests on
                a phrase pattern says Provisional; one that would be a guess says Not observed and
                why.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {s.steps.map((step, i) => (
                <StepCard key={step.name} step={step} i={i} />
              ))}
            </div>
          )}
        </section>
      ))}

      <footer className="border-t border-border pt-6 text-xs leading-relaxed text-muted-foreground">
        Kept in step with the code: constants are named with the file that owns them, and the
        measured examples come from the first full real lesson (17 Aug 2026, period 3). Design
        records: docs/teacher-attribution-plan.md, docs/lesson-coverage-plan.md,
        docs/teacher-measurements.md, docs/rfdetr-pipeline.md.
      </footer>
    </div>
  );
}
