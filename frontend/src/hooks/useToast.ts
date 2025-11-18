import { toast } from 'sonner';

export function useToast() {
  return {
    error: (message: string) => toast.error(message),
    success: (message: string) => toast.success(message),
    info: (message: string) => toast.info(message),
    warning: (message: string) => toast.warning(message),
  };
}
