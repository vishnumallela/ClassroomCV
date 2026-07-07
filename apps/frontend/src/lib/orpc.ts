import { createORPCClient } from "@orpc/client";
import { RPCLink } from "@orpc/client/fetch";
import type { RouterClient } from "@orpc/server";
import { createTanstackQueryUtils } from "@orpc/tanstack-query";
import type { AppRouter } from "@classroom/api-contracts";
import { env } from "@/lib/env";

export const API_URL = env.FRONTEND__API_URL;

const link = new RPCLink({ url: `${API_URL}/rpc` });

export const orpcClient: RouterClient<AppRouter> = createORPCClient(link);
export const orpc = createTanstackQueryUtils(orpcClient);
