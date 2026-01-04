/**
 * T021: AgentConfigPanel component for Oracle agent turn control settings
 * Provides sliders and inputs for configuring agent behavior limits
 */
import { RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import type { AgentConfig } from '@/types/oracle';

interface AgentConfigPanelProps {
  agentConfig: AgentConfig;
  setAgentConfig: (config: AgentConfig) => void;
  isSaving: boolean;
  saved: boolean;
  onReset: () => void;
  isDemoMode?: boolean;
}

/**
 * Format timeout seconds as human-readable duration
 */
function formatTimeout(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins === 0) {
    return `${secs}s`;
  }
  if (secs === 0) {
    return `${mins}m`;
  }
  return `${mins}m ${secs}s`;
}

/**
 * Format token budget with K suffix
 */
function formatTokenBudget(tokens: number): string {
  if (tokens >= 1000) {
    const k = tokens / 1000;
    return k % 1 === 0 ? `${k}K` : `${k.toFixed(1)}K`;
  }
  return tokens.toString();
}

export function AgentConfigPanel({
  agentConfig,
  setAgentConfig,
  isSaving,
  saved,
  onReset,
  isDemoMode = false,
}: AgentConfigPanelProps) {
  const updateConfig = (partial: Partial<AgentConfig>) => {
    setAgentConfig({ ...agentConfig, ...partial });
  };

  return (
    <div className="space-y-6">
      {/* Iteration Limits Group */}
      <div className="space-y-4">
        <h4 className="text-sm font-medium text-muted-foreground">Iteration Limits</h4>

        {/* Max Iterations */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium cursor-help">Max Iterations</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Maximum number of agent turns before forcing completion.
                    Each turn can include multiple tool calls.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className="text-sm text-muted-foreground font-mono">
              {agentConfig.max_iterations}
            </span>
          </div>
          <Slider
            value={[agentConfig.max_iterations]}
            onValueChange={([value]) => updateConfig({ max_iterations: value })}
            min={1}
            max={50}
            step={1}
            disabled={isDemoMode}
          />
          <p className="text-xs text-muted-foreground">
            Range: 1-50 iterations (default: 15)
          </p>
        </div>

        {/* Soft Warning Percent */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium cursor-help">Soft Warning Threshold</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Percentage of max iterations at which the agent receives a warning
                    to wrap up. Helps ensure graceful completion.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className="text-sm text-muted-foreground font-mono">
              {agentConfig.soft_warning_percent}%
            </span>
          </div>
          <Slider
            value={[agentConfig.soft_warning_percent]}
            onValueChange={([value]) => updateConfig({ soft_warning_percent: value })}
            min={50}
            max={90}
            step={5}
            disabled={isDemoMode}
          />
          <p className="text-xs text-muted-foreground">
            Range: 50-90% (default: 70%)
          </p>
        </div>
      </div>

      <Separator />

      {/* Token Budget Group */}
      <div className="space-y-4">
        <h4 className="text-sm font-medium text-muted-foreground">Token Budget</h4>

        {/* Token Budget Input */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium cursor-help">Token Budget</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Maximum tokens the agent can consume across all iterations.
                    Includes both input and output tokens.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className="text-sm text-muted-foreground font-mono">
              {formatTokenBudget(agentConfig.token_budget)}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Input
              type="number"
              min={1000}
              max={200000}
              step={1000}
              value={agentConfig.token_budget}
              onChange={(e) => {
                const value = Math.max(1000, Math.min(200000, parseInt(e.target.value) || 50000));
                updateConfig({ token_budget: value });
              }}
              className="w-32 font-mono"
              disabled={isDemoMode}
            />
            <span className="text-sm text-muted-foreground">tokens (1K-200K)</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Default: 50,000 tokens. Higher budgets allow longer conversations.
          </p>
        </div>

        {/* Token Warning Percent */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium cursor-help">Token Warning Threshold</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Percentage of token budget at which the agent receives a warning.
                    Prompts the agent to be more concise.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className="text-sm text-muted-foreground font-mono">
              {agentConfig.token_warning_percent}%
            </span>
          </div>
          <Slider
            value={[agentConfig.token_warning_percent]}
            onValueChange={([value]) => updateConfig({ token_warning_percent: value })}
            min={50}
            max={95}
            step={5}
            disabled={isDemoMode}
          />
          <p className="text-xs text-muted-foreground">
            Range: 50-95% (default: 80%)
          </p>
        </div>
      </div>

      <Separator />

      {/* Timeouts & Parallelism Group */}
      <div className="space-y-4">
        <h4 className="text-sm font-medium text-muted-foreground">Timeouts & Parallelism</h4>

        {/* Timeout */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium cursor-help">Request Timeout</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Maximum time for the entire agent request.
                    Agent will gracefully terminate if exceeded.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className="text-sm text-muted-foreground font-mono">
              {formatTimeout(agentConfig.timeout_seconds)}
            </span>
          </div>
          <Slider
            value={[agentConfig.timeout_seconds]}
            onValueChange={([value]) => updateConfig({ timeout_seconds: value })}
            min={10}
            max={600}
            step={10}
            disabled={isDemoMode}
          />
          <p className="text-xs text-muted-foreground">
            Range: 10s-10m (default: 2m)
          </p>
        </div>

        {/* Max Tool Calls Per Turn */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium cursor-help">Max Tool Calls Per Turn</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Maximum number of tool calls the agent can make in a single turn.
                    Prevents runaway tool usage.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className="text-sm text-muted-foreground font-mono">
              {agentConfig.max_tool_calls_per_turn}
            </span>
          </div>
          <Slider
            value={[agentConfig.max_tool_calls_per_turn]}
            onValueChange={([value]) => updateConfig({ max_tool_calls_per_turn: value })}
            min={1}
            max={200}
            step={5}
            disabled={isDemoMode}
          />
          <p className="text-xs text-muted-foreground">
            Range: 1-200 calls (default: 100)
          </p>
        </div>

        {/* Max Parallel Tools */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium cursor-help">Max Parallel Tools</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Maximum number of tool calls to execute in parallel.
                    Higher values speed up execution but use more resources.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className="text-sm text-muted-foreground font-mono">
              {agentConfig.max_parallel_tools}
            </span>
          </div>
          <Slider
            value={[agentConfig.max_parallel_tools]}
            onValueChange={([value]) => updateConfig({ max_parallel_tools: value })}
            min={1}
            max={10}
            step={1}
            disabled={isDemoMode}
          />
          <p className="text-xs text-muted-foreground">
            Range: 1-10 parallel (default: 3)
          </p>
        </div>
      </div>

      <Separator />

      {/* Status and Actions */}
      {saved && (
        <Alert>
          <AlertDescription>
            Agent configuration saved successfully!
          </AlertDescription>
        </Alert>
      )}

      <div className="flex items-center justify-between">
        <Button
          variant="outline"
          size="sm"
          onClick={onReset}
          disabled={isDemoMode || isSaving}
        >
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset to Defaults
        </Button>

        {isSaving && (
          <span className="text-sm text-muted-foreground">Saving...</span>
        )}
      </div>

      <p className="text-xs text-muted-foreground">
        These settings control how the Oracle agent manages its execution.
        Adjusting limits can help balance response quality with resource usage.
      </p>
    </div>
  );
}
