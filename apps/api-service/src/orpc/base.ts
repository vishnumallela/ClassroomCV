import { os } from "@orpc/server";

export const base = os.errors({
  NOT_FOUND: { message: "Resource not found." },
  CONFLICT: { message: "Conflict." },
  VALIDATION: { message: "Input validation failed." },
  DEPENDENCY_UNAVAILABLE: { message: "A downstream dependency is unavailable." },
});
