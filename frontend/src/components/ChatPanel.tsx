import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Send, Loader2, Info, Square, GitBranch } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { ChatMessage } from './ChatMessage';
import { SlashCommandMenu } from './SlashCommandMenu';
import { ContextTree } from './ContextTree';
import { streamOracle, cancelOracle, exportConversationAsMarkdown, downloadAsFile, compactHistory } from '@/services/oracle';
import { getModelSettings } from '@/services/models';
import {
  getContextTrees,
  createTree,
  deleteTree,
  checkoutNode,
  labelNode,
  setCheckpoint,
  pruneTree,
  setActiveTree,
} from '@/services/context';
import type { OracleMessage, SlashCommand, OracleStreamChunk, SourceType } from '@/types/oracle';
import type { ModelSettings } from '@/types/models';
import type { ContextTreeData, ContextNode } from '@/types/context';
import { useToast } from '@/hooks/useToast';
import { Badge } from '@/components/ui/badge';

interface ChatPanelProps {
  onNavigateToNote: (path: string) => void;
  onNotesChanged?: () => void;
}

export function ChatPanel({ onNavigateToNote, onNotesChanged }: ChatPanelProps) {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<OracleMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [modelSettings, setModelSettings] = useState<ModelSettings | null>(null);
  const [showThinking, setShowThinking] = useState(true);
  const [showSources, setShowSources] = useState(true);
  const [activeSources, setActiveSources] = useState<SourceType[]>(['vault', 'code', 'threads']);
  const [showCommandMenu, setShowCommandMenu] = useState(false);
  const [commandFilter, setCommandFilter] = useState('');
  const [showContextTree, setShowContextTree] = useState(false);
  const [contextTrees, setContextTrees] = useState<ContextTreeData[]>([]);
  const [activeTreeId, setActiveTreeId] = useState<string | null>(null);
  const [currentContextId, setCurrentContextId] = useState<string | null>(null);
  const [isLoadingTrees, setIsLoadingTrees] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const toast = useToast();

  /**
   * Convert context nodes to OracleMessage format.
   * Builds the path from root to the target node and converts each node
   * to a pair of user/assistant messages.
   */
  const loadMessagesFromTree = useCallback((
    nodes: ContextNode[],
    targetNodeId: string | null
  ): OracleMessage[] => {
    if (!nodes.length || !targetNodeId) {
      return [];
    }

    // Build a map for quick lookup
    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    // Find path from root to target node
    const pathToTarget: ContextNode[] = [];
    let current = nodeMap.get(targetNodeId);

    while (current) {
      pathToTarget.unshift(current);
      current = current.parent_id ? nodeMap.get(current.parent_id) : undefined;
    }

    // Convert nodes to messages (skip root node if it's just a placeholder)
    const messages: OracleMessage[] = [];
    for (const node of pathToTarget) {
      // Skip root nodes that have empty question/answer (placeholder roots)
      if (node.is_root && !node.question && !node.answer) {
        continue;
      }

      // Add user message
      if (node.question) {
        messages.push({
          role: 'user',
          content: node.question,
          timestamp: node.created_at,
        });
      }

      // Add assistant message
      if (node.answer) {
        messages.push({
          role: 'assistant',
          content: node.answer,
          timestamp: node.created_at,
        });
      }
    }

    return messages;
  }, []);

  // Load model settings on mount
  useEffect(() => {
    const loadModelSettings = async () => {
      try {
        const settings = await getModelSettings();
        setModelSettings(settings);
      } catch (err) {
        console.error('Failed to load model settings:', err);
      }
    };
    loadModelSettings();
  }, []);

  // Load context trees on mount and restore messages from active tree
  const loadContextTrees = useCallback(async (restoreMessages = true) => {
    setIsLoadingTrees(true);
    try {
      const response = await getContextTrees();
      setContextTrees(response.trees);
      setActiveTreeId(response.active_tree_id);

      // Restore messages from active tree if available
      if (restoreMessages && response.active_tree_id) {
        const activeTree = response.trees.find(
          (t) => t.tree.root_id === response.active_tree_id
        );
        if (activeTree && activeTree.nodes.length > 0) {
          const restoredMessages = loadMessagesFromTree(
            activeTree.nodes,
            activeTree.tree.current_node_id
          );
          setMessages(restoredMessages);
          setCurrentContextId(activeTree.tree.current_node_id);
          console.debug(
            `Restored ${restoredMessages.length} messages from tree ${response.active_tree_id}`
          );
        }
      } else if (restoreMessages && !response.active_tree_id && response.trees.length > 0) {
        // No active tree but trees exist - auto-activate the most recent one
        // Use the most recent node's created_at time to determine the most recent tree
        const mostRecent = response.trees.sort((a, b) => {
          const aLatest = a.nodes.length > 0
            ? Math.max(...a.nodes.map(n => new Date(n.created_at).getTime()))
            : 0;
          const bLatest = b.nodes.length > 0
            ? Math.max(...b.nodes.map(n => new Date(n.created_at).getTime()))
            : 0;
          return bLatest - aLatest;
        })[0];
        await setActiveTree(mostRecent.tree.root_id);
        // Reload trees after setting active
        return loadContextTrees(true);
      }
    } catch (err) {
      // Context tree API might not be implemented yet - fail silently
      console.debug('Context trees not available:', err);
    } finally {
      setIsLoadingTrees(false);
    }
  }, [loadMessagesFromTree]);

  useEffect(() => {
    loadContextTrees(true); // Restore messages on initial mount
  }, [loadContextTrees]);

  // Context tree handlers
  const handleNewRoot = useCallback(async () => {
    try {
      const tree = await createTree();
      await loadContextTrees(false); // Don't restore messages, we're starting fresh
      setActiveTreeId(tree.root_id);
      setCurrentContextId(null); // Reset context for new tree
      setMessages([]); // Clear messages for new tree
      toast.success('New conversation tree created');
    } catch (err) {
      console.error('Failed to create tree:', err);
      toast.error('Failed to create new conversation tree');
    }
  }, [loadContextTrees, toast]);

  const handleCheckout = useCallback(async (nodeId: string) => {
    try {
      await checkoutNode(nodeId);
      // Reload trees and find the active tree to load messages
      const response = await getContextTrees();
      setContextTrees(response.trees);
      setActiveTreeId(response.active_tree_id);

      // Load messages up to the checked out node
      if (response.active_tree_id) {
        const activeTree = response.trees.find(
          (t) => t.tree.root_id === response.active_tree_id
        );
        if (activeTree) {
          const restoredMessages = loadMessagesFromTree(
            activeTree.nodes,
            nodeId // Load messages up to the checked out node
          );
          setMessages(restoredMessages);
          setCurrentContextId(nodeId);
          console.debug(
            `Checked out to node ${nodeId}, restored ${restoredMessages.length} messages`
          );
        }
      }
      toast.success('Checked out conversation point');
    } catch (err) {
      console.error('Failed to checkout node:', err);
      toast.error('Failed to checkout conversation point');
    }
  }, [loadMessagesFromTree, toast]);

  const handleLabel = useCallback(async (nodeId: string, label: string) => {
    try {
      await labelNode(nodeId, label);
      await loadContextTrees();
      toast.success('Label updated');
    } catch (err) {
      console.error('Failed to label node:', err);
      toast.error('Failed to update label');
    }
  }, [loadContextTrees, toast]);

  const handleCheckpointToggle = useCallback(async (nodeId: string, isCheckpoint: boolean) => {
    try {
      await setCheckpoint(nodeId, isCheckpoint);
      await loadContextTrees();
      toast.success(isCheckpoint ? 'Checkpoint set' : 'Checkpoint removed');
    } catch (err) {
      console.error('Failed to toggle checkpoint:', err);
      toast.error('Failed to update checkpoint');
    }
  }, [loadContextTrees, toast]);

  const handlePrune = useCallback(async (rootId: string) => {
    try {
      const result = await pruneTree(rootId);
      await loadContextTrees();
      toast.success(`Pruned ${result.pruned} nodes`);
    } catch (err) {
      console.error('Failed to prune tree:', err);
      toast.error('Failed to prune tree');
    }
  }, [loadContextTrees, toast]);

  const handleDeleteTree = useCallback(async (rootId: string) => {
    try {
      await deleteTree(rootId);
      await loadContextTrees(false); // Don't restore messages, handled below
      if (activeTreeId === rootId) {
        setMessages([]);
        setCurrentContextId(null);
      }
      toast.success('Conversation tree deleted');
    } catch (err) {
      console.error('Failed to delete tree:', err);
      toast.error('Failed to delete tree');
    }
  }, [loadContextTrees, activeTreeId, toast]);

  const handleSelectTree = useCallback(async (rootId: string) => {
    try {
      await setActiveTree(rootId);
      setActiveTreeId(rootId);

      // Load messages from the selected tree
      const selectedTree = contextTrees.find((t) => t.tree.root_id === rootId);
      if (selectedTree && selectedTree.nodes.length > 0) {
        const restoredMessages = loadMessagesFromTree(
          selectedTree.nodes,
          selectedTree.tree.current_node_id
        );
        setMessages(restoredMessages);
        setCurrentContextId(selectedTree.tree.current_node_id);
        console.debug(
          `Switched to tree ${rootId}, restored ${restoredMessages.length} messages`
        );
      } else {
        setMessages([]);
        setCurrentContextId(null);
      }
      toast.success('Switched to tree');
    } catch (err) {
      console.error('Failed to select tree:', err);
      toast.error('Failed to switch tree');
    }
  }, [contextTrees, loadMessagesFromTree, toast]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  /**
   * Stop the current Oracle query.
   * Aborts the frontend fetch request and cancels the backend session.
   * Defined as useCallback so it can be used in slashCommands.
   */
  const handleStop = useCallback(async () => {
    // Abort the frontend request first (faster feedback)
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Cancel backend session (stops all running agents)
    try {
      await cancelOracle();
    } catch (err) {
      console.error('Failed to cancel Oracle session:', err);
      // Don't show error - frontend abort already worked
    }

    setIsLoading(false);
    setStatusMessage('');
    toast.info('Query stopped');
  }, [toast]);

  // Slash commands
  const slashCommands = useMemo<SlashCommand[]>(
    () => [
      {
        name: 'clear',
        description: 'Clear conversation history',
        shortcut: 'Ctrl+K',
        handler: () => {
          setMessages([]);
          setInput('');
          setCurrentContextId(null); // Reset context for fresh conversation
          toast.success('Conversation cleared');
        },
      },
      {
        name: 'compact',
        description: 'Summarize and compress conversation',
        handler: async () => {
          const compressed = await compactHistory(messages);
          setMessages(compressed);
          toast.success('Conversation compacted');
        },
      },
      {
        name: 'context',
        description: 'Show current context sources being used',
        handler: () => {
          const contextInfo = `Active sources: ${activeSources.join(', ')}`;
          toast.info(contextInfo);
        },
      },
      {
        name: 'help',
        description: 'Show available slash commands',
        handler: () => {
          setShowCommandMenu(true);
          setCommandFilter('');
        },
      },
      {
        name: 'models',
        description: `Configure AI model (current: ${modelSettings?.oracle_model?.split('/').pop() || 'default'})`,
        handler: () => {
          navigate('/settings');
          toast.info('Navigate to Settings → AI Models to change model');
        },
      },
      {
        name: 'sources',
        description: 'Toggle source display on/off',
        handler: () => {
          setShowSources((prev) => !prev);
          toast.success(`Sources ${!showSources ? 'shown' : 'hidden'}`);
        },
      },
      {
        name: 'thinking',
        description: 'Toggle thinking mode display',
        handler: () => {
          setShowThinking((prev) => !prev);
          toast.success(`Thinking ${!showThinking ? 'shown' : 'hidden'}`);
        },
      },
      {
        name: 'export',
        description: 'Export conversation as markdown',
        handler: () => {
          const markdown = exportConversationAsMarkdown(messages);
          const filename = `oracle-conversation-${new Date().toISOString().split('T')[0]}.md`;
          downloadAsFile(markdown, filename);
          toast.success('Conversation exported');
        },
      },
      {
        name: 'stop',
        description: 'Stop current query (if running)',
        shortcut: 'Esc',
        handler: () => {
          if (isLoading) {
            handleStop();
          } else {
            toast.info('No active query to stop');
          }
        },
      },
      {
        name: 'tree',
        description: 'Toggle context tree panel',
        handler: () => {
          setShowContextTree((prev) => !prev);
          toast.success(`Context tree ${!showContextTree ? 'shown' : 'hidden'}`);
        },
      },
      {
        name: 'newbranch',
        description: 'Start a new conversation branch',
        handler: () => {
          handleNewRoot();
        },
      },
    ],
    [messages, activeSources, showSources, showThinking, showContextTree, toast, modelSettings, navigate, isLoading, handleStop, handleNewRoot]
  );

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const trimmedInput = input.trim();

    // Handle slash commands
    if (trimmedInput.startsWith('/')) {
      const commandName = trimmedInput.substring(1).split(' ')[0];
      const command = slashCommands.find((cmd) => cmd.name === commandName);

      if (command) {
        command.handler();
        setInput('');
        setShowCommandMenu(false);
        return;
      }
    }

    const userMsg: OracleMessage = {
      role: 'user',
      content: trimmedInput,
      timestamp: new Date().toISOString(),
    };

    // Add user message
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setStatusMessage('Searching...');

    // Create assistant message placeholder
    const assistantMsg: OracleMessage = {
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      thinking: '',
      sources: [],
    };

    setMessages((prev) => [...prev, assistantMsg]);

    // Create abort controller for this request
    abortControllerRef.current = new AbortController();

    // Track chunks processed for debugging duplication issues
    let chunkProcessedCount = 0;

    try {
      await streamOracle(
        {
          question: trimmedInput,
          sources: activeSources,
          max_results: 10,
          model: modelSettings?.oracle_model,
          thinking: modelSettings?.thinking_enabled,
          context_id: currentContextId ?? undefined, // Pass current context for conversation continuity
        },
        (chunk: OracleStreamChunk) => {
          chunkProcessedCount++;
          // Debug logging to trace chunk duplication
          console.debug(
            `[ChatPanel #${chunkProcessedCount}] Processing chunk type=${chunk.type}`
          );

          setMessages((prev) => {
            // Create a deep copy to avoid mutation issues with React StrictMode
            // StrictMode runs updater functions twice with the same initial state,
            // so in-place mutation causes content to be appended twice.
            const updated = prev.map((msg, idx) =>
              idx === prev.length - 1 ? { ...msg } : msg
            );
            const lastMsg = updated[updated.length - 1];

            if (lastMsg.role === 'assistant') {
              if (chunk.type === 'status') {
                setStatusMessage(chunk.message || '');
              } else if (chunk.type === 'thinking') {
                lastMsg.thinking = (lastMsg.thinking || '') + (chunk.content || '');
              } else if (chunk.type === 'content') {
                lastMsg.content += chunk.content || '';
              } else if (chunk.type === 'source' && chunk.source) {
                // Deep copy sources array to avoid mutation
                lastMsg.sources = [...(lastMsg.sources || []), chunk.source];
              } else if (chunk.type === 'tool_call' && chunk.tool_call) {
                // Add or update tool call in the list
                const existingCalls = lastMsg.tool_calls || [];
                const existingIndex = existingCalls.findIndex(tc => tc.id === chunk.tool_call!.id);
                if (existingIndex >= 0) {
                  // Update existing tool call
                  existingCalls[existingIndex] = {
                    ...existingCalls[existingIndex],
                    ...chunk.tool_call,
                    status: chunk.tool_call.status as 'pending' | 'running' | 'completed' | 'error' || 'running',
                  };
                  lastMsg.tool_calls = [...existingCalls];
                } else {
                  // Add new tool call
                  lastMsg.tool_calls = [...existingCalls, {
                    id: chunk.tool_call.id,
                    name: chunk.tool_call.name,
                    arguments: chunk.tool_call.arguments,
                    status: 'running',
                  }];
                }
                setStatusMessage(`Running ${chunk.tool_call.name}...`);
              } else if (chunk.type === 'tool_result') {
                // Match result to tool call by ID
                const existingCalls = lastMsg.tool_calls || [];
                let matchIndex = -1;

                // First try exact ID match
                if (chunk.tool_call_id) {
                  matchIndex = existingCalls.findIndex(tc => tc.id === chunk.tool_call_id);
                }

                // Fallback: find first running tool without result
                if (matchIndex < 0) {
                  matchIndex = existingCalls.findIndex(tc => !tc.result && tc.status === 'running');
                  if (matchIndex >= 0 && chunk.tool_call_id) {
                    console.warn(
                      `[ChatPanel] Tool result ID mismatch: expected '${chunk.tool_call_id}', ` +
                      `using fallback to running tool '${existingCalls[matchIndex].name}'`
                    );
                  }
                }

                if (matchIndex >= 0) {
                  existingCalls[matchIndex] = {
                    ...existingCalls[matchIndex],
                    result: chunk.tool_result || chunk.content || '',
                    status: 'completed',
                  };
                  lastMsg.tool_calls = [...existingCalls];
                } else {
                  // No matching tool call found - log warning
                  console.warn(
                    `[ChatPanel] Tool result with ID '${chunk.tool_call_id}' has no matching tool call. ` +
                    `Result dropped: ${(chunk.tool_result || '').substring(0, 100)}...`
                  );
                  // Mark message as having a potential issue if tools failed silently
                  if (existingCalls.length > 0) {
                    const hasRunningTools = existingCalls.some(tc => tc.status === 'running');
                    if (!hasRunningTools) {
                      // All tools done but we got an orphan result - might indicate a problem
                      console.warn('[ChatPanel] Orphan tool result received after all tools completed');
                    }
                  }
                }
              } else if (chunk.type === 'done') {
                lastMsg.model = chunk.model_used;
                setStatusMessage('');
                // Mark any remaining running tool calls as completed
                if (lastMsg.tool_calls) {
                  lastMsg.tool_calls = lastMsg.tool_calls.map(tc =>
                    tc.status === 'running' ? { ...tc, status: 'completed' as const } : tc
                  );
                }
                // Save context_id from response for next request
                if (chunk.context_id) {
                  setCurrentContextId(chunk.context_id);
                  console.debug(`Updated context_id to ${chunk.context_id}`);
                }
              } else if (chunk.type === 'error') {
                lastMsg.is_error = true;
                lastMsg.content = chunk.error || 'Unknown error occurred';
              }
            }

            return updated;
          });
        },
        abortControllerRef.current.signal
      );
    } catch (err) {
      // Check if this was a user-initiated abort (stop button)
      if (err instanceof Error && err.name === 'AbortError') {
        // User cancelled - update the message to show it was stopped
        setMessages((prev) => {
          // Deep copy last message to avoid StrictMode mutation issues
          const updated = prev.map((msg, idx) =>
            idx === prev.length - 1 ? { ...msg } : msg
          );
          const lastMsg = updated[updated.length - 1];
          if (lastMsg.role === 'assistant' && !lastMsg.content) {
            lastMsg.content = 'Query stopped by user.';
          }
          return updated;
        });
        return; // Don't show error toast for user-initiated stop
      }

      console.error('Oracle error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to get response';

      setMessages((prev) => {
        // Deep copy last message to avoid StrictMode mutation issues
        const updated = prev.map((msg, idx) =>
          idx === prev.length - 1 ? { ...msg } : msg
        );
        const lastMsg = updated[updated.length - 1];
        if (lastMsg.role === 'assistant') {
          lastMsg.is_error = true;
          lastMsg.content = errorMessage;
        }
        return updated;
      });

      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
      setStatusMessage('');
      abortControllerRef.current = null;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setInput(value);

    // Show command menu when typing slash
    if (value.startsWith('/') && value.length > 1) {
      setShowCommandMenu(true);
      setCommandFilter(value.substring(1));
    } else {
      setShowCommandMenu(false);
      setCommandFilter('');
    }
  };

  const handleCommandSelect = (command: SlashCommand) => {
    command.handler();
    setInput('');
    setShowCommandMenu(false);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-semibold">Vlt Oracle</h2>
            <p className="text-xs text-muted-foreground">
              Multi-source intelligent context retrieval
            </p>
          </div>
          <div className="flex items-center gap-2">
            {modelSettings?.oracle_model && (
              <Badge
                variant="outline"
                className="text-xs cursor-pointer hover:bg-accent"
                onClick={() => navigate('/settings')}
                title="Click to change model"
              >
                {modelSettings.oracle_model.split('/').pop()?.replace(':free', '') || 'default'}
              </Badge>
            )}
            <Button
              variant={showContextTree ? "secondary" : "ghost"}
              size="icon"
              className="h-8 w-8"
              onClick={() => setShowContextTree(!showContextTree)}
              title={showContextTree ? "Hide context tree" : "Show context tree"}
            >
              <GitBranch className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => setShowCommandMenu(true)}
              title="Show commands"
            >
              <Info className="h-4 w-4" />
            </Button>
          </div>
        </div>
        {activeSources.length < 3 && (
          <div className="mt-2 flex gap-1">
            <span className="text-xs text-muted-foreground">Active sources:</span>
            {activeSources.map((source) => (
              <Badge key={source} variant="secondary" className="text-xs">
                {source}
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Context Tree Modal */}
      <Dialog open={showContextTree} onOpenChange={setShowContextTree}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle>Context Trees</DialogTitle>
            <DialogDescription>
              Manage conversation branches and checkpoints
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 overflow-auto">
            <ContextTree
              trees={contextTrees}
              activeTreeId={activeTreeId}
              onCheckout={(nodeId) => {
                handleCheckout(nodeId);
                setShowContextTree(false);
              }}
              onNewRoot={handleNewRoot}
              onLabel={handleLabel}
              onCheckpoint={handleCheckpointToggle}
              onPrune={handlePrune}
              onDeleteTree={handleDeleteTree}
              onSelectTree={(rootId) => {
                handleSelectTree(rootId);
                setShowContextTree(false);
              }}
              isLoading={isLoadingTrees}
            />
          </div>
        </DialogContent>
      </Dialog>

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Message List */}
        <div className="flex-1 overflow-y-auto relative" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-8 text-center">
            <p className="text-base">Ask Oracle anything about your project</p>
            <p className="text-sm mt-2">
              Try: "How does authentication work?" or type <kbd className="px-2 py-1 bg-muted rounded text-xs">/help</kbd>
            </p>
          </div>
        ) : (
          <div className="divide-y divide-border/50">
            {messages.map((msg, i) => (
              <ChatMessage
                key={i}
                message={msg}
                onSourceClick={onNavigateToNote}
                showThinking={showThinking}
                showSources={showSources}
                isStreaming={isLoading && i === messages.length - 1 && msg.role === 'assistant'}
              />
            ))}
            {isLoading && statusMessage && (
              <div className="p-4 flex items-center gap-2 text-muted-foreground text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                {statusMessage}
              </div>
            )}
          </div>
        )}
        </div>
      </div>

      {/* Input Area */}
      <div className="p-4 pb-14 border-t border-border relative">
        {showCommandMenu && (
          <SlashCommandMenu
            commands={slashCommands}
            onSelect={handleCommandSelect}
            onClose={() => setShowCommandMenu(false)}
            filterText={commandFilter}
            position={{ top: -20, left: 0 }}
          />
        )}
        <div className="flex gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question or type / for commands..."
            className="min-h-[40px] max-h-[150px] resize-none"
            rows={1}
            disabled={isLoading}
          />
          {isLoading ? (
            <Button
              onClick={handleStop}
              variant="destructive"
              size="icon"
              title="Stop query"
            >
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button onClick={handleSubmit} disabled={!input.trim()} size="icon">
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            Type <kbd className="px-1 py-0.5 bg-muted rounded">/</kbd> for commands
          </span>
          {messages.length > 0 && (
            <span>{messages.length} messages</span>
          )}
        </div>
      </div>
    </div>
  );
}
