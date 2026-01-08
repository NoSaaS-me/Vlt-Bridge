/**
 * BT Debug Panel - Behavior Tree Debugging Interface
 *
 * Part of the BT Universal Runtime (spec 019).
 * Provides real-time visualization and debugging for behavior trees.
 */

import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, Play, Pause, Trash2, Target, Clock, Database, Bug, ChevronRight, ChevronDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/useToast';
import {
  listTrees,
  getTreeState,
  getBlackboard,
  getTickHistory,
  listBreakpoints,
  setBreakpoint,
  deleteBreakpoint,
  clearAllBreakpoints,
  getTreeVisualization,
  pollTreeState,
} from '@/services/btDebug';
import type {
  TreeSummary,
  TreeStateResponse,
  NodeInfo,
  BlackboardResponse,
  TickHistoryEntry,
  BreakpointInfo,
  TreeStatus,
  RunStatus,
} from '@/types/btDebug';

// =============================================================================
// Status Badge Component
// =============================================================================

interface StatusBadgeProps {
  status: TreeStatus | RunStatus;
}

function StatusBadge({ status }: StatusBadgeProps) {
  const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
    idle: 'secondary',
    running: 'default',
    completed: 'default',
    failed: 'destructive',
    yielded: 'outline',
    fresh: 'secondary',
    success: 'default',
    failure: 'destructive',
  };

  return (
    <Badge variant={variants[status] || 'secondary'}>
      {status.toUpperCase()}
    </Badge>
  );
}

// =============================================================================
// Node Tree Component
// =============================================================================

interface NodeTreeProps {
  node: NodeInfo;
  depth?: number;
  breakpoints: Set<string>;
  onToggleBreakpoint: (nodeId: string) => void;
}

function NodeTree({ node, depth = 0, breakpoints, onToggleBreakpoint }: NodeTreeProps) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = node.children.length > 0;

  return (
    <div className="font-mono text-sm">
      <div
        className={`flex items-center gap-2 py-1 px-2 rounded hover:bg-muted/50 ${
          node.is_active ? 'bg-green-500/20 border-l-2 border-green-500' : ''
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
      >
        {/* Expand/Collapse button */}
        {hasChildren ? (
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-0.5 hover:bg-muted rounded"
          >
            {expanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>
        ) : (
          <span className="w-5" />
        )}

        {/* Breakpoint indicator */}
        <button
          onClick={() => onToggleBreakpoint(node.id)}
          className={`w-3 h-3 rounded-full border-2 ${
            breakpoints.has(node.id)
              ? 'bg-red-500 border-red-500'
              : 'border-muted-foreground/30 hover:border-red-500/50'
          }`}
          title={breakpoints.has(node.id) ? 'Remove breakpoint' : 'Set breakpoint'}
        />

        {/* Node ID */}
        <span className="font-medium">{node.id}</span>

        {/* Node type */}
        <span className="text-muted-foreground">({node.node_type})</span>

        {/* Status */}
        <StatusBadge status={node.status} />

        {/* Timing */}
        {node.last_tick_duration_ms > 0 && (
          <span className="text-muted-foreground text-xs">
            {node.last_tick_duration_ms.toFixed(1)}ms
          </span>
        )}

        {/* Active indicator */}
        {node.is_active && (
          <Badge variant="outline" className="bg-green-500/20 text-green-500 text-xs">
            ACTIVE
          </Badge>
        )}
      </div>

      {/* Children */}
      {expanded && hasChildren && (
        <div>
          {node.children.map((child) => (
            <NodeTree
              key={child.id}
              node={child}
              depth={depth + 1}
              breakpoints={breakpoints}
              onToggleBreakpoint={onToggleBreakpoint}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Tree List Panel
// =============================================================================

interface TreeListPanelProps {
  trees: TreeSummary[];
  selectedTreeId: string | null;
  onSelectTree: (treeId: string) => void;
  onRefresh: () => void;
  isLoading: boolean;
}

function TreeListPanel({ trees, selectedTreeId, onSelectTree, onRefresh, isLoading }: TreeListPanelProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Behavior Trees</CardTitle>
          <Button variant="ghost" size="icon" onClick={onRefresh} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
        <CardDescription>{trees.length} tree(s) registered</CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[200px]">
          {trees.length === 0 ? (
            <p className="text-muted-foreground text-sm text-center py-4">
              No trees loaded
            </p>
          ) : (
            <div className="space-y-2">
              {trees.map((tree) => (
                <button
                  key={tree.id}
                  onClick={() => onSelectTree(tree.id)}
                  className={`w-full text-left p-2 rounded border ${
                    selectedTreeId === tree.id
                      ? 'border-primary bg-primary/5'
                      : 'border-transparent hover:bg-muted/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-sm">{tree.name}</span>
                    <StatusBadge status={tree.status} />
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {tree.node_count} nodes | {tree.tick_count} ticks
                  </div>
                </button>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Blackboard Panel
// =============================================================================

interface BlackboardPanelProps {
  blackboard: BlackboardResponse | null;
  isLoading: boolean;
}

function BlackboardPanel({ blackboard, isLoading }: BlackboardPanelProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        Loading blackboard...
      </div>
    );
  }

  if (!blackboard) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        Select a tree to view blackboard
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 p-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Blackboard State</h3>
          <span className="text-xs text-muted-foreground">
            {(blackboard.size_bytes / 1024).toFixed(1)} KB / {(blackboard.max_size_bytes / 1024 / 1024).toFixed(0)} MB
          </span>
        </div>

        <div className="text-xs text-muted-foreground">
          Scope: {blackboard.scope_name}
          {blackboard.parent_scope && ` (parent: ${blackboard.parent_scope})`}
        </div>

        <Separator />

        <div className="space-y-2">
          {blackboard.keys.map((key) => (
            <div
              key={key.key}
              className={`p-2 rounded border text-sm ${
                blackboard.writes_this_tick.includes(key.key)
                  ? 'border-orange-500/50 bg-orange-500/10'
                  : blackboard.reads_this_tick.includes(key.key)
                  ? 'border-blue-500/50 bg-blue-500/10'
                  : 'border-muted'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono font-medium">{key.key}</span>
                <span className="text-xs text-muted-foreground">{key.schema_type}</span>
              </div>
              {key.has_value && key.value_preview && (
                <pre className="text-xs text-muted-foreground mt-1 overflow-hidden text-ellipsis">
                  {key.value_preview}
                </pre>
              )}
            </div>
          ))}
        </div>
      </div>
    </ScrollArea>
  );
}

// =============================================================================
// History Panel
// =============================================================================

interface HistoryPanelProps {
  treeId: string | null;
}

function HistoryPanel({ treeId }: HistoryPanelProps) {
  const [entries, setEntries] = useState<TickHistoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!treeId) {
      setEntries([]);
      return;
    }

    const loadHistory = async () => {
      setIsLoading(true);
      try {
        const response = await getTickHistory(treeId, { limit: 50 });
        setEntries(response.entries);
      } catch (error) {
        console.error('Failed to load history:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadHistory();
  }, [treeId]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        Loading history...
      </div>
    );
  }

  if (!treeId) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        Select a tree to view history
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-1 p-4">
        {entries.length === 0 ? (
          <p className="text-muted-foreground text-sm text-center">No tick history</p>
        ) : (
          entries.map((entry, i) => (
            <div
              key={`${entry.tick_number}-${i}`}
              className="flex items-center gap-2 text-xs py-1 px-2 rounded hover:bg-muted/50"
            >
              <span className="w-8 text-muted-foreground">#{entry.tick_number}</span>
              <span className="font-mono flex-1">{entry.node_id}</span>
              <StatusBadge status={entry.status} />
              <span className="text-muted-foreground w-16 text-right">
                {entry.duration_ms.toFixed(1)}ms
              </span>
            </div>
          ))
        )}
      </div>
    </ScrollArea>
  );
}

// =============================================================================
// Main BT Debug Panel
// =============================================================================

interface BTDebugPanelProps {
  className?: string;
}

export function BTDebugPanel({ className }: BTDebugPanelProps) {
  const toast = useToast();

  // State
  const [trees, setTrees] = useState<TreeSummary[]>([]);
  const [selectedTreeId, setSelectedTreeId] = useState<string | null>(null);
  const [treeState, setTreeState] = useState<TreeStateResponse | null>(null);
  const [blackboard, setBlackboard] = useState<BlackboardResponse | null>(null);
  const [breakpoints, setBreakpoints] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);
  const [isPolling, setIsPolling] = useState(false);
  const [asciiVisualization, setAsciiVisualization] = useState<string>('');

  // Load tree list
  const loadTrees = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await listTrees();
      setTrees(response.trees);
    } catch (error) {
      console.error('Failed to load trees:', error);
      toast.error('Failed to load behavior trees');
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  // Load selected tree details
  const loadTreeDetails = useCallback(async (treeId: string) => {
    try {
      const [state, bb, bps, viz] = await Promise.all([
        getTreeState(treeId),
        getBlackboard(treeId).catch(() => null),
        listBreakpoints(treeId),
        getTreeVisualization(treeId, { format: 'ascii' }),
      ]);

      setTreeState(state);
      setBlackboard(bb);
      setBreakpoints(new Set(bps.breakpoints.map((bp) => bp.node_id)));
      setAsciiVisualization(viz.ascii_tree);
    } catch (error) {
      console.error('Failed to load tree details:', error);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadTrees();
  }, [loadTrees]);

  // Load details when tree selected
  useEffect(() => {
    if (selectedTreeId) {
      loadTreeDetails(selectedTreeId);
    }
  }, [selectedTreeId, loadTreeDetails]);

  // Polling for live updates
  useEffect(() => {
    if (!selectedTreeId || !isPolling) return;

    const cleanup = pollTreeState(
      selectedTreeId,
      (state) => {
        setTreeState(state);
      },
      1000,
      (error) => {
        console.error('Polling error:', error);
      }
    );

    return cleanup;
  }, [selectedTreeId, isPolling]);

  // Toggle breakpoint
  const handleToggleBreakpoint = async (nodeId: string) => {
    if (!selectedTreeId) return;

    try {
      if (breakpoints.has(nodeId)) {
        await deleteBreakpoint(selectedTreeId, { node_id: nodeId });
        setBreakpoints((prev) => {
          const next = new Set(prev);
          next.delete(nodeId);
          return next;
        });
        toast.success(`Removed breakpoint from ${nodeId}`);
      } else {
        await setBreakpoint(selectedTreeId, { node_id: nodeId });
        setBreakpoints((prev) => new Set([...prev, nodeId]));
        toast.success(`Set breakpoint on ${nodeId}`);
      }
    } catch (error) {
      toast.error('Failed to update breakpoint');
    }
  };

  // Clear all breakpoints
  const handleClearBreakpoints = async () => {
    if (!selectedTreeId) return;

    try {
      await clearAllBreakpoints(selectedTreeId);
      setBreakpoints(new Set());
      toast.success('Cleared all breakpoints');
    } catch (error) {
      toast.error('Failed to clear breakpoints');
    }
  };

  return (
    <div className={`flex flex-col h-full ${className || ''}`}>
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <Bug className="h-5 w-5" />
          <h2 className="font-semibold">BT Debugger</h2>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={isPolling ? 'default' : 'outline'}
            size="sm"
            onClick={() => setIsPolling(!isPolling)}
          >
            {isPolling ? (
              <>
                <Pause className="h-4 w-4 mr-1" />
                Pause
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-1" />
                Live
              </>
            )}
          </Button>
          {breakpoints.size > 0 && (
            <Button variant="outline" size="sm" onClick={handleClearBreakpoints}>
              <Trash2 className="h-4 w-4 mr-1" />
              Clear ({breakpoints.size})
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: Tree list */}
        <div className="w-64 border-r p-4">
          <TreeListPanel
            trees={trees}
            selectedTreeId={selectedTreeId}
            onSelectTree={setSelectedTreeId}
            onRefresh={loadTrees}
            isLoading={isLoading}
          />
        </div>

        {/* Right: Details */}
        <div className="flex-1 overflow-hidden">
          {treeState ? (
            <Tabs defaultValue="tree" className="h-full flex flex-col">
              <TabsList className="mx-4 mt-2">
                <TabsTrigger value="tree">
                  <Target className="h-4 w-4 mr-1" />
                  Tree
                </TabsTrigger>
                <TabsTrigger value="blackboard">
                  <Database className="h-4 w-4 mr-1" />
                  Blackboard
                </TabsTrigger>
                <TabsTrigger value="history">
                  <Clock className="h-4 w-4 mr-1" />
                  History
                </TabsTrigger>
                <TabsTrigger value="ascii">
                  ASCII
                </TabsTrigger>
              </TabsList>

              <TabsContent value="tree" className="flex-1 overflow-hidden m-0">
                <ScrollArea className="h-full">
                  <div className="p-4">
                    <div className="flex items-center gap-4 mb-4">
                      <h3 className="font-semibold text-lg">{treeState.name}</h3>
                      <StatusBadge status={treeState.status} />
                      <span className="text-sm text-muted-foreground">
                        Tick #{treeState.tick_count}
                      </span>
                    </div>
                    <NodeTree
                      node={treeState.root}
                      breakpoints={breakpoints}
                      onToggleBreakpoint={handleToggleBreakpoint}
                    />
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="blackboard" className="flex-1 overflow-hidden m-0">
                <BlackboardPanel blackboard={blackboard} isLoading={false} />
              </TabsContent>

              <TabsContent value="history" className="flex-1 overflow-hidden m-0">
                <HistoryPanel treeId={selectedTreeId} />
              </TabsContent>

              <TabsContent value="ascii" className="flex-1 overflow-hidden m-0">
                <ScrollArea className="h-full">
                  <pre className="p-4 font-mono text-sm whitespace-pre">
                    {asciiVisualization || 'No visualization available'}
                  </pre>
                </ScrollArea>
              </TabsContent>
            </Tabs>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              Select a tree to view details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default BTDebugPanel;
