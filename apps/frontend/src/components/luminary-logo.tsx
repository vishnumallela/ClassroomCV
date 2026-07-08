import { cn } from "@/lib/utils";

/**
 * The Luminary mark: a chalk-white stroke (a piece of chalk, a person at the
 * board, the "l" of Luminary) with one warm shaft of light glowing above it —
 * illumination over teaching, set in a spruce-ink chalkboard squircle. Reads
 * cleanly down to 16px. Logo colors are fixed (not themed) so the mark is
 * identical everywhere; only the optional load-time bloom is animated.
 */
export function LuminaryMark({
  className,
  bloom = false,
  title = "Luminary",
}: {
  className?: string;
  bloom?: boolean;
  title?: string;
}) {
  return (
    <svg
      viewBox="0 0 32 32"
      className={cn("size-8 shrink-0", className)}
      role="img"
      aria-label={title}
    >
      <rect width="32" height="32" rx="8.6" fill="oklch(0.27 0.03 167)" />
      {/* soft amber halo — the light bloom */}
      <circle
        cx="16"
        cy="7.4"
        r="5.6"
        fill="oklch(0.8 0.135 74)"
        opacity="0.24"
        className={bloom ? "bloom" : undefined}
        style={{ transformOrigin: "16px 7.4px" }}
      />
      {/* the chalk stroke */}
      <rect x="13.7" y="11.6" width="4.6" height="14" rx="2.3" fill="oklch(0.97 0.006 150)" />
      {/* the luminary — the warm light */}
      <circle
        cx="16"
        cy="7.4"
        r="3"
        fill="oklch(0.8 0.135 74)"
        className={bloom ? "bloom" : undefined}
        style={{ transformOrigin: "16px 7.4px" }}
      />
    </svg>
  );
}

/**
 * Full lockup: mark + "Luminary" wordmark set in Fraunces. `collapsed` renders
 * the mark alone (for a rail sidebar or small header).
 */
export function LuminaryLogo({
  className,
  collapsed = false,
  bloom = false,
  markClassName,
}: {
  className?: string;
  collapsed?: boolean;
  bloom?: boolean;
  markClassName?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <LuminaryMark bloom={bloom} className={markClassName} />
      {!collapsed && (
        <span className="font-display text-[1.35rem] font-medium leading-none tracking-[-0.02em]">
          Luminary
        </span>
      )}
    </span>
  );
}
