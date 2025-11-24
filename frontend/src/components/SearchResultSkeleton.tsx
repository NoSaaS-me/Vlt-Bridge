import { Skeleton } from "@/components/ui/skeleton";

export function SearchResultSkeleton() {
  return (
    <div className="space-y-2">
      {[1, 2, 3].map((i) => (
        <div key={i} className="p-3 border border-border rounded-md space-y-2">
          {/* Result title */}
          <Skeleton className="h-4 w-2/3" />
          {/* Result path */}
          <Skeleton className="h-3 w-1/2" />
          {/* Result snippet */}
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-4/5" />
        </div>
      ))}
    </div>
  );
}
