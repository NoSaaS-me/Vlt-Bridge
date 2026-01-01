/**
 * T109, T120: Settings page with user profile, API token, and index health
 * Extended with AI model selection for Oracle and Subagents
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Copy, RefreshCw, Check, Save } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, SelectGroup, SelectLabel } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { SettingsSectionSkeleton } from '@/components/SettingsSectionSkeleton';
import { getCurrentUser, getToken, logout, getStoredToken, isDemoSession, AUTH_TOKEN_CHANGED_EVENT } from '@/services/auth';
import { getIndexHealth, rebuildIndex, type RebuildResponse } from '@/services/api';
import { getModels, getModelSettings, saveModelSettings } from '@/services/models';
import { getContextSettings, updateContextSettings } from '@/services/context';
import type { User } from '@/types/user';
import type { IndexHealth } from '@/types/search';
import type { ModelInfo, ModelSettings } from '@/types/models';
import type { ContextSettings } from '@/types/context';
import { SystemLogs } from '@/components/SystemLogs';

export function Settings() {
  const navigate = useNavigate();
  const [user, setUser] = useState<User | null>(null);
  const [apiToken, setApiToken] = useState<string>('');
  const [indexHealth, setIndexHealth] = useState<IndexHealth | null>(null);
  const [copied, setCopied] = useState(false);
  const [isRebuilding, setIsRebuilding] = useState(false);
  const [rebuildResult, setRebuildResult] = useState<RebuildResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDemoMode, setIsDemoMode] = useState<boolean>(isDemoSession());

  // Model settings state
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [modelSettings, setModelSettings] = useState<ModelSettings | null>(null);
  const [isSavingModels, setIsSavingModels] = useState(false);
  const [modelsSaved, setModelsSaved] = useState(false);

  // Context settings state
  const [contextSettings, setContextSettings] = useState<ContextSettings | null>(null);
  const [isSavingContext, setIsSavingContext] = useState(false);
  const [contextSaved, setContextSaved] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    const handler = () => setIsDemoMode(isDemoSession());
    window.addEventListener(AUTH_TOKEN_CHANGED_EVENT, handler);
    return () => window.removeEventListener(AUTH_TOKEN_CHANGED_EVENT, handler);
  }, []);

  const loadData = async () => {
    try {
      const token = getStoredToken();

      // Handle local-dev-token as a special case
      if (token === 'local-dev-token') {
        setUser({
          user_id: 'demo-user',
          vault_path: '/data/vaults/demo-user',
          created: new Date().toISOString(),
        });
        setApiToken(token);
      } else {
        // Real OAuth user
        const userData = await getCurrentUser().catch(() => null);
        setUser(userData);
        if (token) {
          setApiToken(token);
        }
      }

      // Always try to load index health
      const health = await getIndexHealth().catch(() => null);
      setIndexHealth(health);

      // Load model settings and available models
      try {
        const [models, settings] = await Promise.all([
          getModels(),
          getModelSettings(),
        ]);
        setAvailableModels(models);
        setModelSettings(settings);
      } catch (err) {
        console.error('Error loading model settings:', err);
        // Set defaults if API fails
        setModelSettings({
          oracle_model: 'deepseek/deepseek-chat:free',
          oracle_provider: 'openrouter',
          subagent_model: 'google/gemini-2.0-flash-exp:free',
          subagent_provider: 'openrouter',
          thinking_enabled: false,
          chat_center_mode: false,
          librarian_timeout: 1200,
          openrouter_api_key: null,
          openrouter_api_key_set: false,
        });
      }

      // Load context settings
      try {
        const ctxSettings = await getContextSettings();
        setContextSettings(ctxSettings);
      } catch (err) {
        console.debug('Context settings not available:', err);
        // Set defaults if API fails
        setContextSettings({
          max_context_nodes: 30,
        });
      }
    } catch (err) {
      console.error('Error loading settings:', err);
    }
  };

  const handleGenerateToken = async () => {
    if (isDemoMode) {
      setError('Demo mode is read-only. Sign in to generate new tokens.');
      return;
    }
    try {
      setError(null);
      const tokenResponse = await getToken();
      setApiToken(tokenResponse.token);
    } catch (err) {
      setError('Failed to generate token');
      console.error('Error generating token:', err);
    }
  };

  const handleCopyToken = async () => {
    try {
      await navigator.clipboard.writeText(apiToken);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy token:', err);
    }
  };

  const handleRebuildIndex = async () => {
    if (isDemoMode) {
      setError('Demo mode is read-only. Sign in to rebuild the index.');
      return;
    }
    setIsRebuilding(true);
    setError(null);
    setRebuildResult(null);
    
    try {
      const result = await rebuildIndex();
      setRebuildResult(result);
      // Reload health data
      const health = await getIndexHealth();
      setIndexHealth(health);
    } catch (err) {
      setError('Failed to rebuild index');
      console.error('Error rebuilding index:', err);
    } finally {
      setIsRebuilding(false);
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  const getUserInitials = (userId: string) => {
    return userId.slice(0, 2).toUpperCase();
  };

  const handleSaveModelSettings = async () => {
    if (!modelSettings) return;
    if (isDemoMode) {
      setError('Demo mode is read-only. Sign in to save model settings.');
      return;
    }

    setIsSavingModels(true);
    setError(null);
    setModelsSaved(false);

    try {
      await saveModelSettings(modelSettings);
      setModelsSaved(true);
      setTimeout(() => setModelsSaved(false), 2000);
    } catch (err) {
      setError('Failed to save model settings');
      console.error('Error saving model settings:', err);
    } finally {
      setIsSavingModels(false);
    }
  };

  const handleSaveContextSettings = async () => {
    if (!contextSettings) return;
    if (isDemoMode) {
      setError('Demo mode is read-only. Sign in to save context settings.');
      return;
    }

    setIsSavingContext(true);
    setError(null);
    setContextSaved(false);

    try {
      await updateContextSettings(contextSettings);
      setContextSaved(true);
      setTimeout(() => setContextSaved(false), 2000);
    } catch (err) {
      setError('Failed to save context settings');
      console.error('Error saving context settings:', err);
    } finally {
      setIsSavingContext(false);
    }
  };

  const groupModelsByProvider = (models: ModelInfo[]) => {
    const grouped: Record<string, ModelInfo[]> = {
      openrouter: [],
      google: [],
    };

    models.forEach((model) => {
      if (grouped[model.provider]) {
        grouped[model.provider].push(model);
      }
    });

    return grouped;
  };

  const getModelInfo = (modelId: string): ModelInfo | undefined => {
    return availableModels.find((m) => m.id === modelId);
  };

  const formatContextLength = (contextLength: number): string => {
    if (contextLength >= 1000000) {
      return `${(contextLength / 1000000).toFixed(1)}M`;
    } else if (contextLength >= 1000) {
      return `${(contextLength / 1000).toFixed(0)}K`;
    }
    return contextLength.toString();
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={() => navigate('/')}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <h1 className="text-2xl font-bold">Settings</h1>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        {isDemoMode && (
          <Alert variant="destructive">
            <AlertDescription>
              You are viewing the shared demo vault. Sign in with Hugging Face from the main app to enable token generation and index management.
            </AlertDescription>
          </Alert>
        )}
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Profile */}
        {user ? (
          <Card>
            <CardHeader>
              <CardTitle>Profile</CardTitle>
              <CardDescription>Your account information</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <Avatar className="h-16 w-16">
                  <AvatarImage src={user.hf_profile?.avatar_url} />
                  <AvatarFallback>{getUserInitials(user.user_id)}</AvatarFallback>
                </Avatar>
                <div className="flex-1">
                  <div className="font-semibold text-lg">
                    {user.hf_profile?.name || user.hf_profile?.username || user.user_id}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    User ID: {user.user_id}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Vault: {user.vault_path}
                  </div>
                </div>
                <Button variant="outline" onClick={logout}>
                  Sign Out
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <SettingsSectionSkeleton
            title="Profile"
            description="Your account information"
          />
        )}

        {/* API Token */}
        <Card>
          <CardHeader>
            <CardTitle>API Token for MCP</CardTitle>
            <CardDescription>
              Use this token to configure MCP clients (Claude Desktop, etc.)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Bearer Token</label>
              <div className="flex gap-2">
                <Input
                  type="password"
                  value={apiToken}
                  readOnly
                  className="font-mono text-xs"
                  placeholder="Generate a token to get started"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleCopyToken}
                  disabled={!apiToken}
                  title="Copy token"
                >
                  {copied ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            <Button onClick={handleGenerateToken} disabled={isDemoMode}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Generate New Token
            </Button>

            <div className="text-xs text-muted-foreground mt-4">
              <p className="font-semibold mb-2">MCP Configuration (Hosted HTTP):</p>
              <pre className="bg-muted p-3 rounded overflow-x-auto">
{`{
  "mcpServers": {
    "obsidian-docs": {
      "transport": "http",
      "url": "${window.location.origin}/mcp",
      "headers": {
        "Authorization": "Bearer ${apiToken || 'YOUR_TOKEN_HERE'}"
      }
    }
  }
}`}
              </pre>
              <p className="font-semibold mb-2 mt-4">Local Development (STDIO):</p>
              <pre className="bg-muted p-3 rounded overflow-x-auto">
{`{
  "mcpServers": {
    "obsidian-docs": {
      "command": "python",
      "args": ["-m", "backend.src.mcp.server"],
      "cwd": "/absolute/path/to/Document-MCP",
      "env": {
        "LOCAL_USER_ID": "local-dev",
        "PYTHONPATH": "/absolute/path/to/Document-MCP",
        "FASTMCP_SHOW_CLI_BANNER": "false"
      }
    }
  }
}`}
              </pre>
              <p className="text-xs text-muted-foreground mt-2">
                Replace <code className="bg-muted px-1 rounded">/absolute/path/to/Document-MCP</code> with your local checkout path
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Index Health */}
        {indexHealth ? (
          <Card>
            <CardHeader>
              <CardTitle>Index Health</CardTitle>
              <CardDescription>
                Full-text search index status and maintenance
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">Notes Indexed</div>
                  <div className="text-2xl font-bold">{indexHealth.note_count}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Last Updated</div>
                  <div className="text-sm">{formatDate(indexHealth.last_incremental_update)}</div>
                </div>
              </div>

              <Separator />

              <div>
                <div className="text-sm text-muted-foreground mb-1">Last Full Rebuild</div>
                <div className="text-sm">{formatDate(indexHealth.last_full_rebuild)}</div>
              </div>

              {rebuildResult && (
                <Alert>
                  <AlertDescription>
                    Index rebuilt successfully! Indexed {rebuildResult.notes_indexed} notes in {rebuildResult.duration_ms}ms
                  </AlertDescription>
                </Alert>
              )}

              <Button
                onClick={handleRebuildIndex}
                disabled={isDemoMode || isRebuilding}
                variant="outline"
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${isRebuilding ? 'animate-spin' : ''}`} />
                {isRebuilding ? 'Rebuilding...' : 'Rebuild Index'}
              </Button>

              <div className="text-xs text-muted-foreground">
                Rebuilding the index will re-scan all notes and update the full-text search database.
                This may take a few seconds for large vaults.
              </div>
            </CardContent>
          </Card>
        ) : (
          <SettingsSectionSkeleton
            title="Index Health"
            description="Full-text search index status and maintenance"
          />
        )}

        {/* AI Models */}
        {modelSettings ? (
          <Card>
            <CardHeader>
              <CardTitle>AI Models</CardTitle>
              <CardDescription>
                Configure AI models for Oracle and Subagent operations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Oracle Model */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Oracle Model</label>
                <p className="text-xs text-muted-foreground mb-2">
                  Primary model for answering questions and synthesizing context
                </p>
                <Select
                  value={modelSettings.oracle_model}
                  onValueChange={(value) => setModelSettings({ ...modelSettings, oracle_model: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(groupModelsByProvider(availableModels)).map(([provider, models]) => (
                      models.length > 0 && (
                        <SelectGroup key={provider}>
                          <SelectLabel className="capitalize">{provider}</SelectLabel>
                          {models.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              <div className="flex items-center gap-2">
                                <span>{model.name}</span>
                                {model.is_free && (
                                  <Badge variant="secondary" className="text-xs">FREE</Badge>
                                )}
                              </div>
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      )
                    ))}
                  </SelectContent>
                </Select>
                {getModelInfo(modelSettings.oracle_model) && (
                  <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                    <span>Context: {formatContextLength(getModelInfo(modelSettings.oracle_model)!.context_length!)}</span>
                    {getModelInfo(modelSettings.oracle_model)!.is_free && (
                      <Badge variant="outline" className="text-xs">Free Tier</Badge>
                    )}
                  </div>
                )}
              </div>

              <Separator />

              {/* Subagent Model */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Subagent Model</label>
                <p className="text-xs text-muted-foreground mb-2">
                  Model for parallel research and code analysis tasks
                </p>
                <Select
                  value={modelSettings.subagent_model}
                  onValueChange={(value) => setModelSettings({ ...modelSettings, subagent_model: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(groupModelsByProvider(availableModels)).map(([provider, models]) => (
                      models.length > 0 && (
                        <SelectGroup key={provider}>
                          <SelectLabel className="capitalize">{provider}</SelectLabel>
                          {models.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              <div className="flex items-center gap-2">
                                <span>{model.name}</span>
                                {model.is_free && (
                                  <Badge variant="secondary" className="text-xs">FREE</Badge>
                                )}
                              </div>
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      )
                    ))}
                  </SelectContent>
                </Select>
                {getModelInfo(modelSettings.subagent_model) && (
                  <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                    <span>Context: {formatContextLength(getModelInfo(modelSettings.subagent_model)!.context_length!)}</span>
                    {getModelInfo(modelSettings.subagent_model)!.is_free && (
                      <Badge variant="outline" className="text-xs">Free Tier</Badge>
                    )}
                  </div>
                )}
              </div>

              <Separator />

              {/* Thinking Mode */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <label className="text-sm font-medium">Extended Thinking Mode</label>
                    <p className="text-xs text-muted-foreground">
                      Enable deeper reasoning for complex queries (uses more tokens)
                    </p>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div>
                          <Switch
                            checked={modelSettings.thinking_enabled}
                            onCheckedChange={(checked) =>
                              setModelSettings({ ...modelSettings, thinking_enabled: checked })
                            }
                          />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">
                          When enabled, the Oracle will spend more time reasoning through complex
                          questions before providing an answer. This increases accuracy but uses
                          more tokens and takes longer.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>

              <Separator />

              {/* Chat Center Mode */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <label className="text-sm font-medium">Chat Center View</label>
                    <p className="text-xs text-muted-foreground">
                      Show AI chat in center view instead of flyout panel
                    </p>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div>
                          <Switch
                            checked={modelSettings.chat_center_mode}
                            onCheckedChange={(checked) =>
                              setModelSettings({ ...modelSettings, chat_center_mode: checked })
                            }
                          />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">
                          When enabled, clicking the chat button will display the AI chat
                          in the main center panel instead of a side flyout panel.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>

              <Separator />

              {/* Librarian Timeout */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Librarian Timeout</label>
                <p className="text-xs text-muted-foreground mb-2">
                  Maximum time (in minutes) for Librarian subagent operations like summarization and web research
                </p>
                <div className="flex items-center gap-4">
                  <Input
                    type="number"
                    min={1}
                    max={60}
                    value={Math.round(modelSettings.librarian_timeout / 60)}
                    onChange={(e) => {
                      const minutes = Math.max(1, Math.min(60, parseInt(e.target.value) || 20));
                      setModelSettings({ ...modelSettings, librarian_timeout: minutes * 60 });
                    }}
                    className="w-24"
                  />
                  <span className="text-sm text-muted-foreground">minutes (1-60)</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Default: 20 minutes. Increase for large summarization or extensive web research tasks.
                </p>
              </div>

              <Separator />

              {/* OpenRouter API Key */}
              <div className="space-y-2">
                <label className="text-sm font-medium">OpenRouter API Key</label>
                <p className="text-xs text-muted-foreground mb-2">
                  Required for paid models. Get your key at{' '}
                  <a
                    href="https://openrouter.ai/keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    openrouter.ai/keys
                  </a>
                </p>
                <div className="flex gap-2">
                  <Input
                    type="password"
                    placeholder={modelSettings.openrouter_api_key_set ? '••••••••••••••••' : 'sk-or-...'}
                    value={modelSettings.openrouter_api_key || ''}
                    onChange={(e) =>
                      setModelSettings({ ...modelSettings, openrouter_api_key: e.target.value })
                    }
                    className="font-mono text-xs"
                  />
                  {modelSettings.openrouter_api_key_set && !modelSettings.openrouter_api_key && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setModelSettings({ ...modelSettings, openrouter_api_key: '' })}
                      title="Clear saved API key"
                    >
                      Clear
                    </Button>
                  )}
                </div>
                {modelSettings.openrouter_api_key_set && (
                  <p className="text-xs text-green-600 dark:text-green-400">
                    ✓ API key is configured
                  </p>
                )}
              </div>

              {modelsSaved && (
                <Alert>
                  <AlertDescription>
                    Model settings saved successfully!
                  </AlertDescription>
                </Alert>
              )}

              <Button
                onClick={handleSaveModelSettings}
                disabled={isDemoMode || isSavingModels}
              >
                <Save className="h-4 w-4 mr-2" />
                {isSavingModels ? 'Saving...' : 'Save Model Settings'}
              </Button>

              <div className="text-xs text-muted-foreground">
                These settings control which AI models are used for Oracle queries and subagent operations.
                Your API key is stored securely and never exposed in responses.
              </div>
            </CardContent>
          </Card>
        ) : (
          <SettingsSectionSkeleton
            title="AI Models"
            description="Configure AI models for Oracle and Subagent operations"
          />
        )}

        {/* Context Settings */}
        {contextSettings ? (
          <Card>
            <CardHeader>
              <CardTitle>Context Tree Settings</CardTitle>
              <CardDescription>
                Configure Oracle conversation tree behavior
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Max Context Nodes */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Context Nodes</label>
                <p className="text-xs text-muted-foreground mb-2">
                  Maximum conversation nodes to keep per context tree. Older non-checkpoint nodes
                  will be pruned when this limit is approached.
                </p>
                <div className="flex items-center gap-4">
                  <Input
                    type="number"
                    min={5}
                    max={100}
                    value={contextSettings.max_context_nodes}
                    onChange={(e) => {
                      const value = Math.max(5, Math.min(100, parseInt(e.target.value) || 30));
                      setContextSettings({ ...contextSettings, max_context_nodes: value });
                    }}
                    className="w-24"
                  />
                  <span className="text-sm text-muted-foreground">nodes (5-100)</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Default: 30 nodes. Higher values preserve more conversation history but use more storage.
                  Checkpointed nodes are never automatically pruned.
                </p>
              </div>

              {contextSaved && (
                <Alert>
                  <AlertDescription>
                    Context settings saved successfully!
                  </AlertDescription>
                </Alert>
              )}

              <Button
                onClick={handleSaveContextSettings}
                disabled={isDemoMode || isSavingContext}
              >
                <Save className="h-4 w-4 mr-2" />
                {isSavingContext ? 'Saving...' : 'Save Context Settings'}
              </Button>
            </CardContent>
          </Card>
        ) : (
          <SettingsSectionSkeleton
            title="Context Tree Settings"
            description="Configure Oracle conversation tree behavior"
          />
        )}

        {/* System Logs */}
        <SystemLogs />
      </div>
    </div>
  );
}

