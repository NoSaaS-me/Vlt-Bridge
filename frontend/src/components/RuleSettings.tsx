/**
 * RuleSettings component (T069-T071, T088)
 *
 * Displays a list of all rules with their enabled/disabled status,
 * toggle switches for non-core rules, test buttons for demo users,
 * and a plugin section showing installed plugins.
 */

import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, Play, Lock, ChevronDown, ChevronRight, Package, Settings } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { fetchRules, toggleRule, testRule, fetchPlugins } from '@/services/rules';
import type { RuleInfo, RuleTestResponse, HookPoint, PluginInfo } from '@/types/rules';
import { HOOK_POINT_LABELS } from '@/types/rules';

interface RuleSettingsProps {
  isDemoMode: boolean;
  canTestRules?: boolean;
}

interface GroupedRules {
  [key: string]: RuleInfo[];
}

export function RuleSettings({ isDemoMode, canTestRules = false }: RuleSettingsProps) {
  const [rules, setRules] = useState<RuleInfo[]>([]);
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingPlugins, setIsLoadingPlugins] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [togglingRule, setTogglingRule] = useState<string | null>(null);
  const [testingRule, setTestingRule] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{ ruleId: string; result: RuleTestResponse } | null>(
    null
  );
  const [expandedGroups, setExpandedGroups] = useState<Set<HookPoint>>(new Set());

  const loadRules = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetchRules();
      setRules(response.rules);
      // Expand all groups by default
      const allTriggers = new Set<HookPoint>(response.rules.map((r) => r.trigger));
      setExpandedGroups(allTriggers);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load rules');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadPlugins = useCallback(async () => {
    setIsLoadingPlugins(true);

    try {
      const response = await fetchPlugins();
      setPlugins(response.plugins);
    } catch (err) {
      console.debug('No plugins loaded:', err);
      setPlugins([]);
    } finally {
      setIsLoadingPlugins(false);
    }
  }, []);

  useEffect(() => {
    loadRules();
    loadPlugins();
  }, [loadRules, loadPlugins]);

  const handleToggle = async (ruleId: string, newEnabled: boolean) => {
    if (isDemoMode) {
      setError('Demo mode is read-only. Sign in to modify rule settings.');
      return;
    }

    setTogglingRule(ruleId);
    setError(null);

    try {
      const updated = await toggleRule(ruleId, newEnabled);
      setRules((prev) => prev.map((r) => (r.id === ruleId ? { ...r, enabled: updated.enabled } : r)));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle rule');
    } finally {
      setTogglingRule(null);
    }
  };

  const handleTest = async (ruleId: string) => {
    setTestingRule(ruleId);
    setTestResult(null);
    setError(null);

    try {
      const result = await testRule(ruleId);
      setTestResult({ ruleId, result });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to test rule');
    } finally {
      setTestingRule(null);
    }
  };

  const toggleGroup = (trigger: HookPoint) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(trigger)) {
        next.delete(trigger);
      } else {
        next.add(trigger);
      }
      return next;
    });
  };

  // Group rules by trigger
  const groupedRules: GroupedRules = rules.reduce((acc, rule) => {
    if (!acc[rule.trigger]) {
      acc[rule.trigger] = [];
    }
    acc[rule.trigger].push(rule);
    return acc;
  }, {} as GroupedRules);

  // Sort triggers for consistent display
  const sortedTriggers = Object.keys(groupedRules).sort() as HookPoint[];

  if (isLoading && isLoadingPlugins) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Rules & Plugins</CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const handleRefresh = () => {
    loadRules();
    loadPlugins();
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Rules & Plugins</CardTitle>
            <CardDescription>
              Configure Oracle agent behavior rules and manage plugins.
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={isLoading || isLoadingPlugins}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading || isLoadingPlugins ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {testResult && (
          <Alert variant={testResult.result.error ? 'destructive' : 'default'}>
            <AlertDescription>
              <div className="space-y-1">
                <p className="font-medium">Test Result for {testResult.ruleId}</p>
                {testResult.result.error ? (
                  <p className="text-sm text-destructive">{testResult.result.error}</p>
                ) : (
                  <>
                    <p className="text-sm">
                      Condition matched: {testResult.result.condition_result ? 'Yes' : 'No'}
                    </p>
                    <p className="text-sm">
                      Action would execute: {testResult.result.action_would_execute ? 'Yes' : 'No'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Evaluation time: {testResult.result.evaluation_time_ms.toFixed(2)}ms
                    </p>
                  </>
                )}
              </div>
            </AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="rules" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="rules" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Rules ({rules.length})
            </TabsTrigger>
            <TabsTrigger value="plugins" className="flex items-center gap-2">
              <Package className="h-4 w-4" />
              Plugins ({plugins.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="rules" className="mt-4">
            {rules.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">No rules found.</div>
            ) : (
              <div className="space-y-3">
                {sortedTriggers.map((trigger) => (
                  <Collapsible
                    key={trigger}
                    open={expandedGroups.has(trigger)}
                    onOpenChange={() => toggleGroup(trigger)}
                  >
                    <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-md hover:bg-muted/50 transition-colors">
                      <div className="flex items-center gap-2">
                        {expandedGroups.has(trigger) ? (
                          <ChevronDown className="h-4 w-4" />
                        ) : (
                          <ChevronRight className="h-4 w-4" />
                        )}
                        <span className="font-medium">{HOOK_POINT_LABELS[trigger]}</span>
                        <Badge variant="secondary" className="text-xs">
                          {groupedRules[trigger].length}
                        </Badge>
                      </div>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="pl-6 space-y-2 mt-2">
                      {groupedRules[trigger].map((rule) => (
                        <div
                          key={rule.id}
                          className="flex items-center justify-between p-3 rounded-md border bg-card"
                        >
                          <div className="flex-1 min-w-0 mr-4">
                            <div className="flex items-center gap-2">
                              <span className="font-medium truncate">{rule.name}</span>
                              {rule.core && (
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger>
                                      <Badge variant="outline" className="text-xs">
                                        <Lock className="h-3 w-3 mr-1" />
                                        Core
                                      </Badge>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p>Core rules cannot be disabled</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              )}
                              {rule.plugin_id && (
                                <Badge variant="secondary" className="text-xs">
                                  {rule.plugin_id}
                                </Badge>
                              )}
                            </div>
                            {rule.description && (
                              <p className="text-sm text-muted-foreground truncate mt-1">
                                {rule.description}
                              </p>
                            )}
                            <div className="flex items-center gap-2 mt-1">
                              <span className="text-xs text-muted-foreground">
                                Priority: {rule.priority}
                              </span>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            {canTestRules && (
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => handleTest(rule.id)}
                                      disabled={testingRule === rule.id}
                                    >
                                      {testingRule === rule.id ? (
                                        <RefreshCw className="h-4 w-4 animate-spin" />
                                      ) : (
                                        <Play className="h-4 w-4" />
                                      )}
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>Test rule condition</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            )}
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div>
                                    <Switch
                                      checked={rule.enabled}
                                      onCheckedChange={(checked) => handleToggle(rule.id, checked)}
                                      disabled={
                                        rule.core || isDemoMode || togglingRule === rule.id
                                      }
                                    />
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent>
                                  {rule.core ? (
                                    <p>Core rules cannot be disabled</p>
                                  ) : isDemoMode ? (
                                    <p>Sign in to modify rule settings</p>
                                  ) : (
                                    <p>{rule.enabled ? 'Disable' : 'Enable'} this rule</p>
                                  )}
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </div>
                        </div>
                      ))}
                    </CollapsibleContent>
                  </Collapsible>
                ))}
              </div>
            )}

            <Separator className="my-4" />

            <div className="text-xs text-muted-foreground space-y-1">
              <p>
                Rules define conditional behaviors that trigger at specific points in the Oracle agent
                lifecycle.
              </p>
              <p>
                Core rules (marked with <Lock className="inline h-3 w-3" />) are essential for proper
                agent operation and cannot be disabled.
              </p>
            </div>
          </TabsContent>

          <TabsContent value="plugins" className="mt-4">
            {plugins.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Package className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="font-medium">No plugins installed</p>
                <p className="text-sm mt-1">
                  Plugins are optional extensions that package related rules together.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {plugins.map((plugin) => (
                  <div
                    key={plugin.id}
                    className="flex items-center justify-between p-4 rounded-md border bg-card"
                  >
                    <div className="flex-1 min-w-0 mr-4">
                      <div className="flex items-center gap-2">
                        <Package className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium">{plugin.name}</span>
                        <Badge variant="outline" className="text-xs">
                          v{plugin.version}
                        </Badge>
                      </div>
                      {plugin.description && (
                        <p className="text-sm text-muted-foreground mt-1">
                          {plugin.description}
                        </p>
                      )}
                      <div className="flex items-center gap-4 mt-2">
                        <span className="text-xs text-muted-foreground">
                          {plugin.rule_count} rule{plugin.rule_count !== 1 ? 's' : ''}
                        </span>
                        <Badge
                          variant={plugin.enabled ? 'default' : 'secondary'}
                          className="text-xs"
                        >
                          {plugin.enabled ? 'Active' : 'Inactive'}
                        </Badge>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <Separator className="my-4" />

            <div className="text-xs text-muted-foreground space-y-1">
              <p>
                Plugins package related rules together with shared configuration and settings.
              </p>
              <p>
                Install plugins by adding directories with a manifest.toml file to the plugins folder.
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default RuleSettings;
