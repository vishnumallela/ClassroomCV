import { defineConfig } from "drizzle-kit";

export default defineConfig({
  dialect: "postgresql",
  schema: "./src/db/schema/index.ts",
  out: "./drizzle",
  dbCredentials: {
    url:
      process.env.API_SERVICE__DATABASE_URL ??
      "postgres://postgres:postgres@localhost:5433/classroom",
  },
  verbose: true,
  strict: true,
});
