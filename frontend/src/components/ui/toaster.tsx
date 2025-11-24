import { Toaster as Sonner } from 'sonner';

export function Toaster() {
  return (
    <Sonner
      position="top-center"
      toastOptions={{
        duration: 8000,
        unstyled: true,
        classNames: {
          toast: 'relative flex items-center gap-3 rounded-lg border border-border bg-background p-4 shadow-lg',
          error: 'border-red-500/50 bg-red-50 dark:bg-red-950',
          success: 'border-green-500/50 bg-green-50 dark:bg-green-950',
          warning: 'border-yellow-500/50 bg-yellow-50 dark:bg-yellow-950',
          info: 'border-blue-500/50 bg-blue-50 dark:bg-blue-950',
        },
      }}
    />
  );
}
