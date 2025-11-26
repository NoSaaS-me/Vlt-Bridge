/**
 * T109, T120: Settings page with user profile, API token, and index health
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Copy, RefreshCw, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Separator } from '@/components/ui/separator';
import { SettingsSectionSkeleton } from '@/components/SettingsSectionSkeleton';
import { getCurrentUser, getToken, logout, getStoredToken, isDemoSession, AUTH_TOKEN_CHANGED_EVENT } from '@/services/auth';
import { getIndexHealth, rebuildIndex, type RebuildResponse } from '@/services/api';
import type { User } from '@/types/user';
import type { IndexHealth } from '@/types/search';

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
      </div>
    </div>
  );
}

