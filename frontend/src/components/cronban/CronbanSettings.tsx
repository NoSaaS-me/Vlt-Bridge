/**
 * CronbanSettings — Gate verifier model configuration panel.
 *
 * Providers:
 *   claude_code — uses local Claude Code helper sessions (no API key needed)
 *   zai         — z.ai API (OpenAI-compatible, requires base URL + key)
 *   openrouter  — OpenRouter (requires API key)
 *   gemini      — Google Gemini (OpenAI-compatible endpoint, requires API key)
 */
import { useState, useEffect, useCallback } from 'react';
import { Save, FlaskConical, CheckCircle2, XCircle, Loader2, Eye, EyeOff, Bot, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import {
  getGateSettings,
  saveGateSettings,
  testGateModel,
  type CronbanGateSettings,
} from '@/services/cronban-api';

// ---------------------------------------------------------------------------
// Provider metadata
// ---------------------------------------------------------------------------
type Provider = CronbanGateSettings['provider'];

const PROVIDER_LABELS: Record<Provider, string> = {
  claude_code: 'Claude Code (Helper Sessions)',
  zai: 'z.ai',
  openrouter: 'OpenRouter',
  gemini: 'Gemini',
};

const PROVIDER_DEFAULTS: Record<Provider, string> = {
  claude_code: 'sonnet',
  zai: 'z-ai/z-1',
  openrouter: 'openai/gpt-4o-mini',
  gemini: 'gemini-2.5-pro',
};

const PROVIDER_MODEL_HINT: Record<Provider, string> = {
  claude_code: 'Claude model: "haiku", "sonnet", or "opus"',
  zai: 'z.ai model ID, e.g. "z-ai/z-1"',
  openrouter: 'OpenRouter model ID, e.g. "openai/gpt-4o-mini" or "x-ai/grok-4.1-fast"',
  gemini: 'Gemini model ID, e.g. "gemini-2.5-pro" or "gemini-2.0-flash"',
};

/** Providers that need a base URL field */
const NEEDS_BASE_URL = new Set<Provider>(['zai']);
/** Providers that need an API key */
const NEEDS_API_KEY = new Set<Provider>(['zai', 'openrouter', 'gemini']);

interface Props {
  projectId?: string;
}

type TestState = 'idle' | 'running' | 'passed' | 'failed';

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function CronbanSettings({ projectId: _projectId }: Props) {
  const [settings, setSettings] = useState<CronbanGateSettings | null>(null);
  const [provider, setProvider] = useState<Provider>('claude_code');
  const [model, setModel] = useState(PROVIDER_DEFAULTS.claude_code);
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [testState, setTestState] = useState<TestState>('idle');
  const [testResult, setTestResult] = useState<{ met: boolean; reasoning: string } | null>(null);

  // ── Load current settings ─────────────────────────────────────────────────
  const loadSettings = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getGateSettings();
      setSettings(data);
      setProvider(data.provider);
      setModel(data.model);
      setBaseUrl(data.base_url ?? '');
    } catch (err) {
      console.error('Failed to load gate settings:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  // ── Provider switch — reset model to provider default ────────────────────
  function handleProviderChange(newProvider: Provider) {
    setProvider(newProvider);
    setModel(PROVIDER_DEFAULTS[newProvider] ?? '');
    setBaseUrl('');
  }

  // ── Save ─────────────────────────────────────────────────────────────────
  async function handleSave() {
    setSaving(true);
    setSaved(false);
    setSaveError(null);
    try {
      const payload: { provider: string; model: string; api_key?: string; base_url?: string } = {
        provider,
        model,
        base_url: baseUrl.trim() || undefined,
      };
      if (apiKey.trim()) payload.api_key = apiKey.trim();
      await saveGateSettings(payload);
      setSaved(true);
      setApiKey('');
      setTimeout(() => setSaved(false), 3000);
      await loadSettings();
    } catch (err: unknown) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  // ── Test model ────────────────────────────────────────────────────────────
  async function handleTest() {
    setTestState('running');
    setTestResult(null);
    try {
      const resp = await testGateModel();
      if (resp.ok && resp.result) {
        setTestState(resp.result.met ? 'passed' : 'failed');
        setTestResult({ met: resp.result.met ?? false, reasoning: resp.result.reasoning ?? '' });
      } else {
        setTestState('failed');
        setTestResult({ met: false, reasoning: 'Unexpected response from test endpoint.' });
      }
    } catch (err: unknown) {
      setTestState('failed');
      setTestResult({ met: false, reasoning: err instanceof Error ? err.message : String(err) });
    }
  }

  const needsApiKey = NEEDS_API_KEY.has(provider);
  const needsBaseUrl = NEEDS_BASE_URL.has(provider);
  const isConfigured = settings?.has_api_key || false;
  const isClaudeCode = provider === 'claude_code';

  // ── Render ────────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 text-muted-foreground text-sm">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading gate settings…
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle>Gate Verifier Model</CardTitle>
            <CardDescription>
              The LLM used to evaluate whether a Cronban gate condition has been met.
              This model sees the hidden eval text — the working agent never does.
            </CardDescription>
          </div>
          <div className="flex items-center gap-1.5">
            <span className={cn('h-2 w-2 rounded-full', isConfigured ? 'bg-green-500' : 'bg-muted-foreground')} />
            <span className="text-xs text-muted-foreground">
              {isConfigured ? 'Configured' : 'Not configured'}
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Provider select */}
        <div className="space-y-1.5">
          <Label htmlFor="gate-provider">Provider</Label>
          <Select value={provider} onValueChange={(v) => handleProviderChange(v as Provider)}>
            <SelectTrigger id="gate-provider" className="w-56">
              <SelectValue placeholder="Select provider" />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(PROVIDER_LABELS) as Provider[]).map((p) => (
                <SelectItem key={p} value={p}>{PROVIDER_LABELS[p]}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Claude Code info box */}
        {isClaudeCode && (
          <div className="rounded-md border border-blue-500/30 bg-blue-500/5 p-3 space-y-1.5">
            <div className="flex items-center gap-2 text-sm font-medium text-blue-400">
              <Bot className="h-4 w-4" />
              Helper Session Pool
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Evaluations run through a pool of persistent Claude Code sessions. Each session
              accumulates project context over time via <code className="font-mono">--resume</code>.
              When all helpers are busy, a new session is spawned automatically. No API key required.
            </p>
            <div className="flex items-start gap-1.5 text-xs text-muted-foreground">
              <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              <span>Set the model below to control which Claude version runs evaluations.</span>
            </div>
          </div>
        )}

        {/* Model name */}
        <div className="space-y-1.5">
          <Label htmlFor="gate-model">Model</Label>
          <Input
            id="gate-model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder={PROVIDER_DEFAULTS[provider]}
            className="font-mono text-sm"
          />
          <p className="text-xs text-muted-foreground">{PROVIDER_MODEL_HINT[provider]}</p>
        </div>

        {/* Base URL (z.ai and other self-hosted) */}
        {needsBaseUrl && (
          <div className="space-y-1.5">
            <Label htmlFor="gate-base-url">API Base URL</Label>
            <Input
              id="gate-base-url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.z.ai/v1"
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              OpenAI-compatible endpoint base URL (without <code className="font-mono">/chat/completions</code>).
            </p>
          </div>
        )}

        {/* API Key (hidden for claude_code) */}
        {needsApiKey && (
          <div className="space-y-1.5">
            <Label htmlFor="gate-api-key">API Key</Label>
            <div className="relative flex items-center">
              <Input
                id="gate-api-key"
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={settings?.has_api_key ? '••••••  (key set — enter new to replace)' : 'Enter API key'}
                className="pr-10 font-mono text-sm"
                autoComplete="off"
              />
              <button
                type="button"
                onClick={() => setShowKey((s) => !s)}
                className="absolute right-2.5 text-muted-foreground hover:text-foreground transition-colors"
                aria-label={showKey ? 'Hide key' : 'Show key'}
              >
                {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <p className="text-xs text-muted-foreground">
              Stored at rest. Leave blank to keep the existing key.
            </p>
          </div>
        )}

        {/* Save / Test buttons */}
        <div className="flex items-center gap-3 pt-1">
          <Button onClick={handleSave} disabled={saving} size="sm">
            {saving ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Save className="h-4 w-4 mr-2" />}
            Save
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={handleTest}
            disabled={testState === 'running' || !isConfigured}
            title={!isConfigured ? 'Configure provider before testing' : undefined}
          >
            {testState === 'running' ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <FlaskConical className="h-4 w-4 mr-2" />
            )}
            Test
          </Button>

          {saved && (
            <Badge variant="outline" className="gap-1 text-green-600 border-green-600">
              <CheckCircle2 className="h-3 w-3" />
              Saved
            </Badge>
          )}
        </div>

        {saveError && (
          <Alert variant="destructive">
            <AlertDescription>{saveError}</AlertDescription>
          </Alert>
        )}

        {testResult && testState !== 'running' && (
          <>
            <Separator />
            <div
              className={cn(
                'rounded-md border p-3 space-y-1',
                testState === 'passed'
                  ? 'border-green-500/40 bg-green-500/5'
                  : 'border-destructive/40 bg-destructive/5',
              )}
            >
              <div className="flex items-center gap-2 text-sm font-medium">
                {testState === 'passed' ? (
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                ) : (
                  <XCircle className="h-4 w-4 text-destructive" />
                )}
                {testState === 'passed' ? 'Test passed' : 'Test failed'}
              </div>
              {testResult.reasoning && (
                <p className="text-xs text-muted-foreground leading-relaxed">{testResult.reasoning}</p>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
