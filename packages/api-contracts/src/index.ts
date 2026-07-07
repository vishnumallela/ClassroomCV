import type { InferRouterInputs, InferRouterOutputs, RouterClient } from "@orpc/server";
import type { AppRouter } from "@classroom/api-service/routers";

export type { AppRouter } from "@classroom/api-service/routers";

export type RouterInputs = InferRouterInputs<AppRouter>;
export type RouterOutputs = InferRouterOutputs<AppRouter>;
export type AppRouterClient = RouterClient<AppRouter>;
