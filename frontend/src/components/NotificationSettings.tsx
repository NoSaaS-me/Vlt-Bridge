/**
 * T064, T067: Notification Settings Component
 * Displays subscriber toggles for notification management
 * Core subscribers are shown as always-enabled with tooltip explanation
 */
import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Skeleton } from '@/components/ui/skeleton';
import { Bell, Lock } from 'lucide-react';
import { getSubscribers, toggleSubscriber } from '@/services/notifications';
import type { SubscriberInfo } from '@/types/notifications';

/**
 * Priority grouping for subscribers
 * Maps event prefixes to priority levels
 */
const PRIORITY_ORDER = ['critical', 'high', 'normal', 'low'] as const;
type Priority = typeof PRIORITY_ORDER[number];

/**
 * Determine priority based on event types
 */
function getSubscriberPriority(subscriber: SubscriberInfo): Priority {
  const events = subscriber.events || [];

  // Core subscribers are typically critical
  if (subscriber.is_core) {
    return 'critical';
  }

  // Check event types for priority hints
  for (const event of events) {
    if (event.includes('exceeded') || event.includes('critical')) {
      return 'critical';
    }
    if (event.includes('failure') || event.includes('timeout') || event.includes('error')) {
      return 'high';
    }
    if (event.includes('warning')) {
      return 'normal';
    }
  }

  return 'normal';
}

/**
 * Get display properties for priority
 */
function getPriorityBadge(priority: Priority): { label: string; variant: 'default' | 'secondary' | 'outline' | 'destructive' } {
  switch (priority) {
    case 'critical':
      return { label: 'Critical', variant: 'destructive' };
    case 'high':
      return { label: 'High', variant: 'default' };
    case 'normal':
      return { label: 'Normal', variant: 'secondary' };
    case 'low':
      return { label: 'Low', variant: 'outline' };
  }
}

interface NotificationSettingsProps {
  isDemoMode?: boolean;
}

export function NotificationSettings({ isDemoMode = false }: NotificationSettingsProps) {
  const [subscribers, setSubscribers] = useState<SubscriberInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [togglingIds, setTogglingIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadSubscribers();
  }, []);

  const loadSubscribers = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSubscribers();
      setSubscribers(data);
    } catch (err) {
      console.error('Error loading subscribers:', err);
      setError('Failed to load notification subscribers');
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = async (subscriberId: string, newEnabled: boolean) => {
    if (isDemoMode) {
      setError('Demo mode is read-only. Sign in to change notification settings.');
      return;
    }

    // Prevent double-toggling
    if (togglingIds.has(subscriberId)) {
      return;
    }

    setTogglingIds((prev) => new Set(prev).add(subscriberId));
    setError(null);

    try {
      const result = await toggleSubscriber(subscriberId, newEnabled);
      // Update local state with new enabled status
      setSubscribers((prev) =>
        prev.map((s) =>
          s.id === subscriberId ? { ...s, enabled: result.enabled } : s
        )
      );
    } catch (err) {
      console.error('Error toggling subscriber:', err);
      setError(`Failed to toggle subscriber: ${subscriberId}`);
    } finally {
      setTogglingIds((prev) => {
        const next = new Set(prev);
        next.delete(subscriberId);
        return next;
      });
    }
  };

  // Group subscribers by priority
  const groupedSubscribers = subscribers.reduce(
    (acc, subscriber) => {
      const priority = getSubscriberPriority(subscriber);
      if (!acc[priority]) {
        acc[priority] = [];
      }
      acc[priority].push(subscriber);
      return acc;
    },
    {} as Record<Priority, SubscriberInfo[]>
  );

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notification Subscribers
          </CardTitle>
          <CardDescription>
            Configure which notifications appear in your agent conversations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex items-center justify-between p-3 border rounded-lg">
              <div className="space-y-2">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-3 w-48" />
              </div>
              <Skeleton className="h-6 w-11" />
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bell className="h-5 w-5" />
          Notification Subscribers
        </CardTitle>
        <CardDescription>
          Configure which notifications appear in your agent conversations.
          Core notifications cannot be disabled as they are essential for agent operation.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {subscribers.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-8">
            No notification subscribers configured.
          </div>
        ) : (
          PRIORITY_ORDER.map((priority) => {
            const prioritySubscribers = groupedSubscribers[priority];
            if (!prioritySubscribers || prioritySubscribers.length === 0) {
              return null;
            }

            const priorityBadge = getPriorityBadge(priority);

            return (
              <div key={priority} className="space-y-3">
                <div className="flex items-center gap-2">
                  <Badge variant={priorityBadge.variant}>{priorityBadge.label}</Badge>
                  <span className="text-sm text-muted-foreground">
                    ({prioritySubscribers.length} subscriber{prioritySubscribers.length !== 1 ? 's' : ''})
                  </span>
                </div>

                <div className="space-y-2">
                  {prioritySubscribers.map((subscriber) => (
                    <SubscriberRow
                      key={subscriber.id}
                      subscriber={subscriber}
                      isToggling={togglingIds.has(subscriber.id)}
                      onToggle={handleToggle}
                      isDemoMode={isDemoMode}
                    />
                  ))}
                </div>
              </div>
            );
          })
        )}

        <div className="text-xs text-muted-foreground pt-4 border-t">
          Notifications help your AI agent stay informed about tool failures, budget limits,
          and other important events during execution.
        </div>
      </CardContent>
    </Card>
  );
}

interface SubscriberRowProps {
  subscriber: SubscriberInfo;
  isToggling: boolean;
  onToggle: (id: string, newEnabled: boolean) => void;
  isDemoMode: boolean;
}

function SubscriberRow({ subscriber, isToggling, onToggle, isDemoMode }: SubscriberRowProps) {
  const isDisabled = subscriber.is_core || isToggling || isDemoMode;

  return (
    <div
      className={`flex items-center justify-between p-3 border rounded-lg ${
        subscriber.enabled ? 'bg-background' : 'bg-muted/50'
      }`}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm truncate">{subscriber.name}</span>
          {subscriber.is_core && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Lock className="h-3.5 w-3.5 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Core notification - cannot be disabled.
                    This notification is essential for agent operation.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          <Badge variant="outline" className="text-xs">
            v{subscriber.version}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground mt-0.5 truncate">
          {subscriber.description}
        </p>
        <div className="flex flex-wrap gap-1 mt-1">
          {subscriber.events.slice(0, 3).map((event) => (
            <Badge key={event} variant="secondary" className="text-xs font-mono">
              {event}
            </Badge>
          ))}
          {subscriber.events.length > 3 && (
            <Badge variant="secondary" className="text-xs">
              +{subscriber.events.length - 3} more
            </Badge>
          )}
        </div>
      </div>

      <div className="ml-4">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div>
                <Switch
                  checked={subscriber.enabled}
                  onCheckedChange={(checked) => onToggle(subscriber.id, checked)}
                  disabled={isDisabled}
                  aria-label={`Toggle ${subscriber.name}`}
                />
              </div>
            </TooltipTrigger>
            {subscriber.is_core && (
              <TooltipContent>
                <p>Core notification, cannot be disabled</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}
