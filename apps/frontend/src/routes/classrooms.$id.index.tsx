import { createFileRoute, redirect } from "@tanstack/react-router";

// A classroom's home is its lessons.
export const Route = createFileRoute("/classrooms/$id/")({
  beforeLoad: ({ params }) => {
    throw redirect({ to: "/classrooms/$id/videos", params });
  },
});
