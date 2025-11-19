import { Skeleton } from "@/components/ui/skeleton";

export function NoteViewerSkeleton() {
  return (
    <div className="space-y-6 p-6">
      {/* Title skeleton */}
      <Skeleton className="h-8 w-3/4" />

      {/* Metadata skeleton */}
      <div className="flex gap-4 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-24" />
        </div>
        <div className="flex items-center gap-2">
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-24" />
        </div>
      </div>

      {/* Divider */}
      <Skeleton className="h-px w-full" />

      {/* Content skeleton - multiple lines */}
      <div className="space-y-3">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
          </div>
        ))}
      </div>

      {/* Backlinks section skeleton */}
      <div className="border-t pt-6">
        <Skeleton className="h-5 w-32 mb-4" />
        <div className="space-y-2">
          {[1, 2].map((i) => (
            <Skeleton key={i} className="h-4 w-full" />
          ))}
        </div>
      </div>
    </div>
  );
}
