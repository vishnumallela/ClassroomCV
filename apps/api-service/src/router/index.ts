import { analysisRouter } from "@api/router/analysis";
import { boardRouter } from "@api/router/board";
import { classroomsRouter } from "@api/router/classrooms";
import { lessonsRouter } from "@api/router/lessons";
import { gpuRouter, settingsRouter } from "@api/router/settings";
import { videosRouter } from "@api/router/videos";
import { zonesRouter } from "@api/router/zones";

export const appRouter = {
  classrooms: classroomsRouter,
  videos: videosRouter,
  zones: zonesRouter,
  board: boardRouter,
  analysis: analysisRouter,
  lessons: lessonsRouter,
  settings: settingsRouter,
  gpu: gpuRouter,
};

export type AppRouter = typeof appRouter;
