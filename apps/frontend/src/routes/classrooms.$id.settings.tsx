import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { PenLine } from "lucide-react";
import { useEffect, useState } from "react";
import { ZoneEditor, type ZonePayload } from "@/components/zone-editor";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { API_URL, orpc, orpcClient } from "@/lib/orpc";

export const Route = createFileRoute("/classrooms/$id/settings")({
  component: ClassroomSettings,
});

const ZONE_LABEL: Record<string, string> = { board: "Board", door: "Door" };

/**
 * Classroom configuration: identity, the zone template every new upload
 * inherits, and deletion. Zones are drawn on a real frame from the room's
 * latest lesson so the polygons line up with what the camera actually sees.
 */
function ClassroomSettings() {
  const { id } = Route.useParams();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { data, isLoading } = useQuery(orpc.classrooms.get.queryOptions({ input: { id } }));

  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [description, setDescription] = useState("");
  const [editorOpen, setEditorOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    if (data) {
      setName(data.name);
      setLocation(data.location ?? "");
      setDescription(data.description ?? "");
    }
  }, [data]);

  const update = useMutation({
    mutationFn: () =>
      orpcClient.classrooms.update({
        id,
        name: name.trim() || undefined,
        location: location.trim() || null,
        description: description.trim() || null,
      }),
    onSuccess: () => queryClient.invalidateQueries(),
  });

  const remove = useMutation({
    mutationFn: () => orpcClient.classrooms.delete({ id }),
    onSuccess: async () => {
      await queryClient.invalidateQueries();
      void navigate({ to: "/" });
    },
    onError: () =>
      setDeleteError("This classroom still holds lessons. Delete them first."),
  });

  if (isLoading || !data) {
    return <Skeleton className="h-64 rounded-xl" />;
  }

  // Zones are drawn over a real frame: the newest lesson that has a thumbnail.
  const reference = data.videos.find((v) => v.thumbnailUrl !== null) ?? null;
  const dirty =
    name.trim() !== data.name ||
    (location.trim() || null) !== (data.location ?? null) ||
    (description.trim() || null) !== (data.description ?? null);

  return (
    <div className="space-y-6">
      <Card className="space-y-4 p-6">
        <div className="space-y-1">
          <h2 className="font-display text-lg font-medium">Details</h2>
          <p className="text-sm text-muted-foreground">
            How this room appears across Luminary.
          </p>
        </div>
        <form
          className="grid gap-4 sm:grid-cols-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (dirty && name.trim()) update.mutate();
          }}
        >
          <label className="block space-y-1.5">
            <span className="text-sm font-medium">Name</span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={120}
              required
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
          </label>
          <label className="block space-y-1.5">
            <span className="text-sm font-medium">Location</span>
            <input
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              maxLength={500}
              placeholder="Block B, Room 12"
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
          </label>
          <label className="block space-y-1.5 sm:col-span-2">
            <span className="text-sm font-medium">Notes</span>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              maxLength={500}
              rows={2}
              placeholder="Camera above the door, board on the left wall…"
              className="w-full resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
          </label>
          <div className="flex items-center gap-3 sm:col-span-2">
            <Button type="submit" size="sm" disabled={!dirty || !name.trim() || update.isPending}>
              {update.isPending ? "Saving…" : "Save details"}
            </Button>
            {update.isSuccess && !dirty && (
              <span className="text-xs text-muted-foreground">Saved.</span>
            )}
          </div>
        </form>
      </Card>

      <Card className="space-y-4 p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <h2 className="font-display text-lg font-medium">Zones</h2>
            <p className="max-w-md text-sm text-muted-foreground">
              The board and door template for this room's camera. Every lesson uploaded from now
              on inherits it, so the GPU never has to guess where the board is.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            disabled={!reference}
            onClick={() => setEditorOpen(true)}
          >
            <PenLine className="size-3.5" />
            {data.zones.length > 0 ? "Edit zones" : "Configure zones"}
          </Button>
        </div>

        {data.zones.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {data.zones.map((z) => (
              <Badge key={z.id} variant="secondary">
                {ZONE_LABEL[z.kind] ?? z.kind} · {z.polygon.length} points
              </Badge>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            {reference
              ? "No zones configured yet. New lessons will fall back to automatic board/door detection."
              : "Upload one lesson first — zones are drawn on a real frame from this room's camera."}
          </p>
        )}
      </Card>

      <Card className="space-y-3 border-destructive/40 p-6">
        <div className="space-y-1">
          <h2 className="font-display text-lg font-medium">Danger zone</h2>
          <p className="text-sm text-muted-foreground">
            Deleting a classroom is permanent. Its lessons must be deleted first — nothing is
            removed silently.
          </p>
        </div>
        {confirmDelete ? (
          <div className="flex items-center gap-2">
            <Button
              variant="destructive"
              size="sm"
              disabled={remove.isPending}
              onClick={() => remove.mutate()}
            >
              {remove.isPending ? "Deleting…" : "Yes, delete this classroom"}
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
          </div>
        ) : (
          <Button variant="outline" size="sm" onClick={() => setConfirmDelete(true)}>
            Delete classroom
          </Button>
        )}
        {deleteError && <p className="text-xs text-destructive">{deleteError}</p>}
      </Card>

      {editorOpen && reference && (
        <ZoneEditor
          videoId={reference.id}
          title="Configure classroom zones"
          frameSrc={reference.thumbnailUrl ? `${API_URL}${reference.thumbnailUrl}` : null}
          aspect={16 / 9}
          initialZones={data.zones.map((z) => ({ kind: z.kind, polygon: z.polygon }))}
          onSave={async (zones: ZonePayload[]) => {
            await orpcClient.classrooms.setZones({ id, zones });
          }}
          onClose={() => setEditorOpen(false)}
        />
      )}
    </div>
  );
}
