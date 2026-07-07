import { analysisRouter } from "@api/router/analysis";
import { boardRouter } from "@api/router/board";
import { videosRouter } from "@api/router/videos";
import { zonesRouter } from "@api/router/zones";

export const appRouter = {
  videos: videosRouter,
  zones: zonesRouter,
  board: boardRouter,
  analysis: analysisRouter,
};

export type AppRouter = typeof appRouter;
