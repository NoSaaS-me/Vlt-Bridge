import { Skeleton } from "@/components/ui/skeleton";

export function DirectoryTreeSkeleton() {
  return (
    <div className="space-y-3 p-4">
      {/* Main folders */}
      {[1, 2, 3].map((i) => (
        <div key={i} className="space-y-2">
          {/* Folder item */}
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4" />
            <Skeleton className="h-4 w-32" />
          </div>

          {/* Sub-items */}
          {[1, 2].map((j) => (
            <div key={j} className="ml-4 flex items-center gap-2">
              <Skeleton className="h-4 w-4" />
              <Skeleton className="h-4 w-28" />
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
