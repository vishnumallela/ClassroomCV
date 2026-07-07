import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import { env } from "@api/lib/env";
import * as schema from "@api/db/schema";

export const sql = postgres(env.API_SERVICE__DATABASE_URL, {
  max: 10,
  idle_timeout: 20,
  connect_timeout: 10,
});

export const db = drizzle(sql, { schema });

export async function pingDb(): Promise<void> {
  await sql`select 1`;
}

export async function closeDb(): Promise<void> {
  await sql.end({ timeout: 5 });
}
