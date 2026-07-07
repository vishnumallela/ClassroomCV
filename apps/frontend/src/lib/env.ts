import { createEnv } from "@t3-oss/env-core";
import * as z from "zod";

export const env = createEnv({
  clientPrefix: "FRONTEND__",
  client: {
    FRONTEND__API_URL: z.url().default("http://localhost:8787"),
  },
  runtimeEnv: import.meta.env,
  emptyStringAsUndefined: true,
});
