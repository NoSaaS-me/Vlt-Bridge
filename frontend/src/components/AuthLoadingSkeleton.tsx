import { Skeleton } from "@/components/ui/skeleton";

export function AuthLoadingSkeleton() {
  return (
    <div className="h-screen flex flex-col items-center justify-center bg-background">
      <div className="space-y-6 max-w-md w-full px-6">
        {/* Logo/header area */}
        <div className="flex flex-col items-center gap-4">
          <Skeleton className="h-12 w-12 rounded-full" />
          <Skeleton className="h-6 w-48" />
        </div>

        {/* Loading animation indicator */}
        <div className="flex items-center justify-center gap-2">
          <div className="h-2 w-2 bg-muted rounded-full animate-skeleton-pulse" />
          <div className="h-2 w-2 bg-muted rounded-full animate-skeleton-pulse" style={{ animationDelay: '0.4s' }} />
          <div className="h-2 w-2 bg-muted rounded-full animate-skeleton-pulse" style={{ animationDelay: '0.8s' }} />
        </div>

        {/* Subtle loading text */}
        <div className="text-center">
          <Skeleton className="h-4 w-32 mx-auto" />
        </div>
      </div>
    </div>
  );
}
