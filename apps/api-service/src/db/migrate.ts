import { migrate } from "drizzle-orm/postgres-js/migrator";
import { closeDb, db } from "@api/lib/db";
import { logger } from "@api/lib/logger";

export async function runMigrations(): Promise<void> {
  await migrate(db, { migrationsFolder: "./drizzle" });
}

if (import.meta.main) {
  runMigrations()
    .then(closeDb)
    .then(() => process.exit(0))
    .catch((err) => {
      logger.error({ err }, "migration failed");
      process.exit(1);
    });
}
