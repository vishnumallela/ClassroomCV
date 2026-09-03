/**
 * Backfill `videos.recording_started_at` from each recording's own container.
 *
 * The probe step fills this in going forward, but every lesson analysed before
 * migration 0012 has it null — and without it nothing in Group A of
 * docs/teacher-measurements.md can be computed, so an entire archive reports
 * "Not Observed" until someone types 39 timestamps by hand. The files already
 * know: 36 of 39 sample recordings carry a creation_time tag.
 *
 *   bun apps/api-service/scripts/backfill-recording-start.ts            # dry run
 *   bun apps/api-service/scripts/backfill-recording-start.ts --apply    # write
 *
 * Dry run by default. `--apply` is required to touch a row, and a row that
 * already has a value is never overwritten — a typed-in correction outranks a
 * container tag, which is exactly the case where the tag was wrong.
 */
import { and, eq, isNull } from "drizzle-orm";
import { videos } from "@api/db/schema";
import { db } from "@api/lib/db";
import { probeVideo } from "@api/lib/media";
import { localDateInSchoolTz, schoolTimezone } from "@api/lib/school-time";
import { isS3, presignGet } from "@api/lib/storage";

/**
 * Local file first, presigned URL second.
 *
 * This script runs on the machine that holds the videos, so the local copy is
 * free and instant. It also covers the lessons uploaded before the 2026-07-22
 * switch to MinIO, whose bytes were never written to the object store at all —
 * going straight to a presigned URL fails on every one of those.
 */
async function source(filePath: string): Promise<string> {
  if (await Bun.file(filePath).exists()) return filePath;
  return isS3 ? (presignGet(filePath, 60 * 60) ?? filePath) : filePath;
}

async function main(): Promise<void> {
  const apply = process.argv.includes("--apply");
  const tz = await schoolTimezone();

  const rows = await db
    .select({
      id: videos.id,
      title: videos.title,
      filePath: videos.filePath,
      lessonDate: videos.lessonDate,
    })
    .from(videos)
    .where(isNull(videos.recordingStartedAt));

  console.log(`${rows.length} lesson(s) with no recording start. Timezone: ${tz}.`);
  console.log(apply ? "Applying.\n" : "Dry run — pass --apply to write.\n");

  let found = 0;
  let missing = 0;
  let failed = 0;

  for (const row of rows) {
    let startedAt: Date | null = null;
    try {
      startedAt = (await probeVideo(await source(row.filePath))).recordingStartedAt;
    } catch (err) {
      failed++;
      console.log(`  ✗ ${row.title} — unreadable (${err instanceof Error ? err.message : err})`);
      continue;
    }

    if (!startedAt) {
      missing++;
      console.log(`  – ${row.title} — no creation_time in the container; type it in`);
      continue;
    }

    found++;
    const localDate = localDateInSchoolTz(startedAt, tz);
    console.log(`  ✓ ${row.title} — ${startedAt.toISOString()} (${localDate} local)`);

    if (apply) {
      await db
        .update(videos)
        .set({
          recordingStartedAt: startedAt,
          // Only when nobody has set one: the date is a fact about the lesson,
          // and the recording's own day is a guess at it, not an authority.
          ...(row.lessonDate === null ? { lessonDate: localDate } : {}),
        })
        // Keyed on the row id AND still-null, so a concurrent write (an
        // analysis finishing, someone typing one in) is never clobbered.
        .where(and(eq(videos.id, row.id), isNull(videos.recordingStartedAt)));
    }
  }

  console.log(
    `\n${found} anchored, ${missing} without a tag, ${failed} unreadable.` +
      (apply ? "" : "\nNothing written — re-run with --apply."),
  );
}

await main();
process.exit(0);
