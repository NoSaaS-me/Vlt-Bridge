/**
 * ContextTree Component
 * Timeline-based visual display for Oracle conversation branches
 *
 * Design Principles:
 * 1. Linear sections are FLAT - No indentation for parent->single-child chains
 * 2. Only indent at branch points - Where a node has multiple children
 * 3. Collapse long chains - Show "... (N more)" instead of every node
 * 4. Branch points are visually distinct - Fork icon and styling
 * 5. HEAD is prominent - Clearly marked and easy to find
 */
import { useState, useMemo, useCallback } from 'react';
import {
  Star,
  GitBranch,
  Plus,
  Scissors,
  Tag,
  Trash2,
  MoreVertical,
  GitFork,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import type { ContextNode, ContextTreeData } from '@/types/context';

// ============================================================================
// Types
// ============================================================================

export interface ContextTreeProps {
  trees: ContextTreeData[];
  activeTreeId: string | null;
  onCheckout: (nodeId: string) => void;
  onNewRoot: () => void;
  onLabel: (nodeId: string, label: string) => void;
  onCheckpoint: (nodeId: string, isCheckpoint: boolean) => void;
  onPrune: (rootId: string) => void;
  onDeleteTree: (rootId: string) => void;
  onSelectTree: (rootId: string) => void;
  isLoading?: boolean;
}

type TimelineNodeType = 'linear' | 'branch' | 'head';

interface TimelineNode {
  node: ContextNode;
  type: TimelineNodeType;
  children: TimelineNode[];
  isHead: boolean;
  isBranchPoint: boolean;
  depth: number;
}

interface CollapsedChain {
  start: TimelineNode;
  hidden: TimelineNode[];
  end: TimelineNode;
}

interface TimelineSegment {
  type: 'node' | 'collapsed';
  node?: TimelineNode;
  collapsed?: CollapsedChain;
  depth: number;
  isBranchChild?: boolean;
  isLastBranchChild?: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Build a map of parent_id -> children
 */
function buildChildrenMap(nodes: ContextNode[]): Map<string | null, ContextNode[]> {
  const map = new Map<string | null, ContextNode[]>();

  for (const node of nodes) {
    const parentId = node.parent_id;
    if (!map.has(parentId)) {
      map.set(parentId, []);
    }
    map.get(parentId)!.push(node);
  }

  // Sort children by created_at
  for (const [, children] of map) {
    children.sort((a, b) =>
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );
  }

  return map;
}

/**
 * Get all ancestor node IDs for a given node (path from root to node)
 */
function getPathToNode(nodeId: string, nodes: ContextNode[]): Set<string> {
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const path = new Set<string>();

  let current = nodeMap.get(nodeId);
  while (current) {
    path.add(current.id);
    current = current.parent_id ? nodeMap.get(current.parent_id) : undefined;
  }

  return path;
}

/**
 * Build the timeline tree structure from flat nodes
 */
function buildTimelineTree(
  nodes: ContextNode[],
  currentNodeId: string
): TimelineNode | null {
  if (nodes.length === 0) return null;

  const childrenMap = buildChildrenMap(nodes);
  const rootNode = nodes.find(n => n.is_root);
  if (!rootNode) return null;

  function buildNode(node: ContextNode, depth: number): TimelineNode {
    const nodeChildren = childrenMap.get(node.id) || [];
    const isHead = node.id === currentNodeId;
    const isBranchPoint = nodeChildren.length > 1;

    const type: TimelineNodeType = isHead ? 'head' :
                                   isBranchPoint ? 'branch' : 'linear';

    return {
      node,
      type,
      isHead,
      isBranchPoint,
      depth,
      children: nodeChildren.map(child =>
        buildNode(child, isBranchPoint ? depth + 1 : depth)
      ),
    };
  }

  return buildNode(rootNode, 0);
}

/**
 * Flatten the timeline tree into segments, collapsing long linear chains
 */
function flattenTimeline(
  root: TimelineNode,
  _pathToHead: Set<string>, // Available for future use (e.g., never collapse path to HEAD)
  expandedChains: Set<string>,
  collapseThreshold: number = 3
): TimelineSegment[] {
  const segments: TimelineSegment[] = [];

  function processLinearChain(nodes: TimelineNode[]): TimelineSegment[] {
    const result: TimelineSegment[] = [];

    // Check if we should collapse this chain
    const chainId = nodes[0]?.node.id;
    const isExpanded = expandedChains.has(chainId);
    const shouldCollapse = !isExpanded && nodes.length > collapseThreshold;

    // Don't collapse if HEAD is in the chain
    const headInChain = nodes.some(n => n.isHead);

    if (shouldCollapse && !headInChain) {
      // Show: first -> "... (N more)" -> last
      result.push({ type: 'node', node: nodes[0], depth: nodes[0].depth });

      if (nodes.length > 2) {
        result.push({
          type: 'collapsed',
          collapsed: {
            start: nodes[0],
            hidden: nodes.slice(1, -1),
            end: nodes[nodes.length - 1],
          },
          depth: nodes[0].depth,
        });
      }

      result.push({
        type: 'node',
        node: nodes[nodes.length - 1],
        depth: nodes[nodes.length - 1].depth
      });
    } else {
      // Show all nodes
      for (const node of nodes) {
        result.push({ type: 'node', node, depth: node.depth });
      }
    }

    return result;
  }

  function walk(node: TimelineNode, isBranchChild: boolean = false, isLastBranchChild: boolean = false): void {
    // Collect linear chain
    const linearChain: TimelineNode[] = [node];
    let current = node;

    while (
      current.children.length === 1 &&
      !current.isBranchPoint &&
      !current.isHead
    ) {
      current = current.children[0];
      linearChain.push(current);
    }

    // Process the linear chain
    const chainSegments = processLinearChain(linearChain);

    // Mark first segment as branch child if applicable
    if (chainSegments.length > 0 && isBranchChild) {
      chainSegments[0].isBranchChild = true;
      chainSegments[0].isLastBranchChild = isLastBranchChild;
    }

    segments.push(...chainSegments);

    // Get the last node in the chain for processing children
    const lastInChain = linearChain[linearChain.length - 1];

    // Process children of the last node in the chain
    if (lastInChain.isBranchPoint) {
      // Branch point: process each child as a branch child
      lastInChain.children.forEach((child, idx) => {
        walk(child, true, idx === lastInChain.children.length - 1);
      });
    } else if (lastInChain.children.length === 1 && lastInChain.isHead) {
      // HEAD has a child - continue walking
      walk(lastInChain.children[0]);
    }
  }

  walk(root);
  return segments;
}

/**
 * Truncate text for display
 */
function truncateText(text: string, maxLength: number = 40): string {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
}

// ============================================================================
// Timeline Node Component
// ============================================================================

interface TimelineNodeRowProps {
  segment: TimelineSegment;
  pathToHead: Set<string>;
  onCheckout: (nodeId: string) => void;
  onLabel: (nodeId: string, label: string) => void;
  onCheckpoint: (nodeId: string, isCheckpoint: boolean) => void;
  onExpandChain: (chainId: string) => void;
}

function TimelineNodeRow({
  segment,
  pathToHead,
  onCheckout,
  onLabel,
  onCheckpoint,
  onExpandChain,
}: TimelineNodeRowProps) {
  const [isLabelDialogOpen, setIsLabelDialogOpen] = useState(false);
  const [labelInput, setLabelInput] = useState('');

  if (segment.type === 'collapsed') {
    const collapsed = segment.collapsed!;
    const hiddenCount = collapsed.hidden.length;

    return (
      <div
        className="flex items-center pl-4 py-0.5"
        style={{ marginLeft: `${segment.depth * 16}px` }}
      >
        <span className="text-muted-foreground mr-2">|</span>
        <button
          onClick={() => onExpandChain(collapsed.start.node.id)}
          className="text-xs text-muted-foreground hover:text-foreground hover:underline cursor-pointer"
        >
          ... ({hiddenCount} more)
        </button>
      </div>
    );
  }

  // After the collapsed check above, segment.node is guaranteed to exist
  const timelineNode = segment.node!;
  const { node } = timelineNode;
  const isOnPath = pathToHead.has(node.id);
  const displayText = node.label || truncateText(node.question || 'Root');

  const handleLabelSubmit = () => {
    onLabel(node.id, labelInput);
    setIsLabelDialogOpen(false);
  };

  const openLabelDialog = () => {
    setLabelInput(node.label || '');
    setIsLabelDialogOpen(true);
  };

  // Determine connector characters
  let connector = '';
  let connectorClass = 'text-muted-foreground/50';

  if (segment.isBranchChild) {
    connector = segment.isLastBranchChild ? 'L' : 'T';
    connectorClass = 'text-muted-foreground/70';
  } else if (timelineNode.isBranchPoint) {
    connector = 'F'; // Fork
    connectorClass = 'text-amber-500';
  } else {
    connector = '|';
  }

  return (
    <>
      <div
        className={cn(
          "group flex items-center py-0.5 hover:bg-accent/30 rounded transition-colors cursor-pointer",
          timelineNode.isHead && "bg-primary/10 hover:bg-primary/15",
          isOnPath && !timelineNode.isHead && "bg-accent/20"
        )}
        style={{ paddingLeft: `${segment.depth * 16 + 4}px` }}
        onClick={() => onCheckout(node.id)}
      >
        {/* Timeline connector */}
        <span className={cn("w-4 text-center font-mono text-xs", connectorClass)}>
          {connector === 'T' && <span>|-</span>}
          {connector === 'L' && <span>'-</span>}
          {connector === 'F' && <GitFork className="h-3 w-3 inline" />}
          {connector === '|' && <span>|</span>}
        </span>

        {/* Node indicator */}
        <span className="w-4 text-center mx-0.5">
          {timelineNode.isHead ? (
            <span className="inline-block w-2 h-2 rounded-full bg-primary" />
          ) : (
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-muted-foreground/40" />
          )}
        </span>

        {/* Checkpoint star */}
        {node.is_checkpoint && (
          <Star className="h-3 w-3 text-yellow-500 fill-yellow-500 mr-1 shrink-0" />
        )}

        {/* Node text */}
        <span className={cn(
          "flex-1 text-sm truncate",
          timelineNode.isHead && "font-medium"
        )}>
          {displayText}
        </span>

        {/* HEAD badge */}
        {timelineNode.isHead && (
          <Badge variant="default" className="text-[10px] py-0 px-1 h-4 ml-1 shrink-0">
            HEAD
          </Badge>
        )}

        {/* Branch indicator */}
        {timelineNode.isBranchPoint && (
          <Badge variant="outline" className="text-[10px] py-0 px-1 h-4 ml-1 shrink-0 text-amber-600">
            {timelineNode.children.length}
          </Badge>
        )}

        {/* Context menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
            <Button
              variant="ghost"
              size="icon"
              className="h-5 w-5 p-0 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
            >
              <MoreVertical className="h-3 w-3" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-44">
            <DropdownMenuItem onClick={() => onCheckout(node.id)}>
              <GitBranch className="h-4 w-4 mr-2" />
              Checkout
            </DropdownMenuItem>
            <DropdownMenuItem onClick={openLabelDialog}>
              <Tag className="h-4 w-4 mr-2" />
              {node.label ? 'Edit Label' : 'Add Label'}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => onCheckpoint(node.id, !node.is_checkpoint)}>
              <Star className={cn("h-4 w-4 mr-2", node.is_checkpoint && "fill-yellow-500")} />
              {node.is_checkpoint ? 'Remove Checkpoint' : 'Set Checkpoint'}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Label dialog */}
      <Dialog open={isLabelDialogOpen} onOpenChange={setIsLabelDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Label Node</DialogTitle>
            <DialogDescription>
              Add a descriptive label to this conversation point.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Input
              placeholder="e.g., Setup complete, Working approach"
              value={labelInput}
              onChange={(e) => setLabelInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleLabelSubmit()}
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsLabelDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleLabelSubmit}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

// ============================================================================
// Timeline Display Component
// ============================================================================

interface TimelineDisplayProps {
  treeData: ContextTreeData;
  isActive: boolean;
  onCheckout: (nodeId: string) => void;
  onLabel: (nodeId: string, label: string) => void;
  onCheckpoint: (nodeId: string, isCheckpoint: boolean) => void;
  onPrune: (rootId: string) => void;
  onDeleteTree: (rootId: string) => void;
  onSelectTree: (rootId: string) => void;
}

function TimelineDisplay({
  treeData,
  isActive,
  onCheckout,
  onLabel,
  onCheckpoint,
  onPrune,
  onDeleteTree,
  onSelectTree,
}: TimelineDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(isActive);
  const [expandedChains, setExpandedChains] = useState<Set<string>>(new Set());

  const { tree, nodes } = treeData;

  // Build timeline structure
  const timelineRoot = useMemo(
    () => buildTimelineTree(nodes, tree.current_node_id),
    [nodes, tree.current_node_id]
  );

  // Get path to HEAD for highlighting
  const pathToHead = useMemo(
    () => getPathToNode(tree.current_node_id, nodes),
    [tree.current_node_id, nodes]
  );

  // Flatten into segments
  const segments = useMemo(() => {
    if (!timelineRoot) return [];
    return flattenTimeline(timelineRoot, pathToHead, expandedChains);
  }, [timelineRoot, pathToHead, expandedChains]);

  const handleExpandChain = useCallback((chainId: string) => {
    setExpandedChains(prev => {
      const next = new Set(prev);
      next.add(chainId);
      return next;
    });
  }, []);

  const warningLevel = tree.node_count / tree.max_nodes;
  const showWarning = warningLevel >= 0.8;

  return (
    <div className={cn(
      "border rounded-lg mb-2",
      isActive ? "border-primary/50 bg-accent/10" : "border-border"
    )}>
      {/* Tree header */}
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <div className="flex items-center justify-between p-2">
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="flex-1 justify-start gap-2 h-7 px-2">
              {isExpanded ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              <GitBranch className="h-3 w-3" />
              <span className="font-medium text-sm truncate">
                {tree.label || `Tree ${tree.root_id.slice(0, 6)}`}
              </span>
              {isActive && (
                <Badge variant="default" className="text-[10px] py-0 px-1 h-4">
                  Active
                </Badge>
              )}
            </Button>
          </CollapsibleTrigger>

          <div className="flex items-center gap-1">
            {/* Node count */}
            <Badge
              variant={showWarning ? "destructive" : "secondary"}
              className="text-[10px] px-1"
            >
              {tree.node_count}/{tree.max_nodes}
            </Badge>

            {/* Tree actions */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-6 w-6">
                  <MoreVertical className="h-3 w-3" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {!isActive && (
                  <DropdownMenuItem onClick={() => onSelectTree(tree.root_id)}>
                    <GitBranch className="h-4 w-4 mr-2" />
                    Set Active
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem onClick={() => onPrune(tree.root_id)}>
                  <Scissors className="h-4 w-4 mr-2" />
                  Prune Old Nodes
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => onDeleteTree(tree.root_id)}
                  className="text-destructive"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete Tree
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Warning banner */}
        {showWarning && isExpanded && (
          <div className="px-2 pb-1">
            <div className="text-[10px] text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30 rounded px-2 py-0.5">
              Approaching limit. Consider pruning.
            </div>
          </div>
        )}

        {/* Timeline nodes */}
        <CollapsibleContent>
          <div className="pb-2">
            {segments.length === 0 ? (
              <div className="text-sm text-muted-foreground text-center py-4">
                No conversation nodes yet
              </div>
            ) : (
              segments.map((segment) => (
                <TimelineNodeRow
                  key={segment.type === 'collapsed'
                    ? `collapsed-${segment.collapsed!.start.node.id}`
                    : segment.node!.node.id
                  }
                  segment={segment}
                  pathToHead={pathToHead}
                  onCheckout={onCheckout}
                  onLabel={onLabel}
                  onCheckpoint={onCheckpoint}
                  onExpandChain={handleExpandChain}
                />
              ))
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Main ContextTree Component
// ============================================================================

export function ContextTree({
  trees,
  activeTreeId,
  onCheckout,
  onNewRoot,
  onLabel,
  onCheckpoint,
  onPrune,
  onDeleteTree,
  onSelectTree,
  isLoading,
}: ContextTreeProps) {
  // Sort trees: active first, then by most recent
  const sortedTrees = useMemo(() => {
    return [...trees].sort((a, b) => {
      if (a.tree.root_id === activeTreeId) return -1;
      if (b.tree.root_id === activeTreeId) return 1;
      // Sort by most recent node activity
      const aLatest = Math.max(...a.nodes.map(n => new Date(n.created_at).getTime()), 0);
      const bLatest = Math.max(...b.nodes.map(n => new Date(n.created_at).getTime()), 0);
      return bLatest - aLatest;
    });
  }, [trees, activeTreeId]);

  if (isLoading) {
    return (
      <div className="p-4">
        <div className="animate-pulse space-y-2">
          <div className="h-6 bg-muted rounded w-3/4" />
          <div className="h-16 bg-muted rounded" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-2 border-b border-border">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-sm">Context</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={onNewRoot}
            className="h-6 text-xs px-2"
          >
            <Plus className="h-3 w-3 mr-1" />
            New
          </Button>
        </div>
      </div>

      {/* Tree list */}
      <ScrollArea className="flex-1">
        <div className="p-2">
          {sortedTrees.length === 0 ? (
            <div className="text-center py-6 text-muted-foreground">
              <GitBranch className="h-6 w-6 mx-auto mb-2 opacity-50" />
              <p className="text-xs">No context trees yet</p>
              <p className="text-[10px] mt-0.5">Start a conversation to create one</p>
            </div>
          ) : (
            sortedTrees.map((treeData) => (
              <TimelineDisplay
                key={treeData.tree.root_id}
                treeData={treeData}
                isActive={treeData.tree.root_id === activeTreeId}
                onCheckout={onCheckout}
                onLabel={onLabel}
                onCheckpoint={onCheckpoint}
                onPrune={onPrune}
                onDeleteTree={onDeleteTree}
                onSelectTree={onSelectTree}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
