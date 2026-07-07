import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { API_URL } from "@/lib/orpc";
import { cn } from "@/lib/utils";

export function UploadZone() {
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const upload = (file: File) => {
    setError(null);
    setProgress(0);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/videos?filename=${encodeURIComponent(file.name)}`);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) setProgress(e.loaded / e.total);
    });
    xhr.addEventListener("load", () => {
      setProgress(null);
      if (xhr.status >= 200 && xhr.status < 300) {
        void queryClient.invalidateQueries();
      } else {
        setError(`Upload failed (${xhr.status})`);
      }
    });
    xhr.addEventListener("error", () => {
      setProgress(null);
      setError("Upload failed. Is the analytics service running?");
    });
    xhr.send(file);
  };

  const onFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) upload(file);
  };

  return (
    <Card
      className={cn(
        "border-dashed p-8 text-center transition-colors",
        dragging && "border-primary bg-primary/5",
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        onFiles(e.dataTransfer.files);
      }}
    >
      {progress !== null ? (
        <div className="space-y-2">
          <div className="text-sm text-muted-foreground">
            Uploading {Math.round(progress * 100)}%
          </div>
          <div className="mx-auto h-1.5 w-64 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      ) : (
        <>
          <p className="text-sm">Drop a classroom recording to analyze</p>
          <p className="mt-1 text-xs text-muted-foreground">MP4, WebM, MOV, or MKV</p>
          <Button className="mt-4" size="sm" onClick={() => inputRef.current?.click()}>
            Choose file
          </Button>
          {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
        </>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => onFiles(e.target.files)}
      />
    </Card>
  );
}
