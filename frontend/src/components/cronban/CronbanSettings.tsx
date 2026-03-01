/**
 * CronbanSettings — Gate verifier model configuration panel.
 *
 * Lets the user pick a provider (OpenRouter | Gemini), configure a model
 * name, store an API key, and test the connection with a dummy verify call.
 */
import { useState, useEffect, useCallback } from 'react';
import { Save, FlaskConical, CheckCircle2, XCircle, Loader2, Eye, EyeOff } from 'lucide-react';
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
// Defaults per provider
// ---------------------------------------------------------------------------
const PROVIDER_DEFAULTS: Record<string, string> = {
  openrouter: 'openai/gpt-4o-mini',
  gemini: 'gemini-2.5-pro',
};

interface Props {
  projectId?: string;
}

type TestState = 'idle' | 'running' | 'passed' | 'failed';

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function CronbanSettings({ projectId: _projectId }: Props) {
  const [settings, setSettings] = useState<CronbanGateSettings | null>(null);
  const [provider, setProvider] = useState<'openrouter' | 'gemini'>('openrouter');
  const [model, setModel] = useState(PROVIDER_DEFAULTS.openrouter);
  const [apiKey, setApiKey] = useState('');
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
      // Never pre-fill the API key field — the placeholder shows "••••••" if set
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
  function handleProviderChange(newProvider: 'openrouter' | 'gemini') {
    setProvider(newProvider);
    setModel(PROVIDER_DEFAULTS[newProvider] ?? '');
  }

  // ── Save ─────────────────────────────────────────────────────────────────
  async function handleSave() {
    setSaving(true);
    setSaved(false);
    setSaveError(null);
    try {
      const payload: { provider: string; model: string; api_key?: string } = {
        provider,
        model,
      };
      // Only send api_key if user actually typed something
      if (apiKey.trim()) {
        payload.api_key = apiKey.trim();
      }
      await saveGateSettings(payload);
      setSaved(true);
      setApiKey(''); // Clear field after save (treat as write-only)
      setTimeout(() => setSaved(false), 3000);
      await loadSettings(); // Refresh has_api_key indicator
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
        setTestResult({
          met: resp.result.met ?? false,
          reasoning: resp.result.reasoning ?? '',
        });
      } else {
        setTestState('failed');
        setTestResult({ met: false, reasoning: 'Unexpected response from test endpoint.' });
      }
    } catch (err: unknown) {
      setTestState('failed');
      setTestResult({
        met: false,
        reasoning: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // ── Status badge ──────────────────────────────────────────────────────────
  const isConfigured = settings?.has_api_key || false;

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
              This model sees the hidden eval text — Claude never does.
            </CardDescription>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className={cn(
                'h-2 w-2 rounded-full',
                isConfigured ? 'bg-green-500' : 'bg-muted-foreground',
              )}
            />
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
          <Select value={provider} onValueChange={(v) => handleProviderChange(v as 'openrouter' | 'gemini')}>
            <SelectTrigger id="gate-provider" className="w-48">
              <SelectValue placeholder="Select provider" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="openrouter">OpenRouter</SelectItem>
              <SelectItem value="gemini">Gemini</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Model name */}
        <div className="space-y-1.5">
          <Label htmlFor="gate-model">Model</Label>
          <Input
            id="gate-model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder={PROVIDER_DEFAULTS[provider] ?? 'model-name'}
            className="font-mono text-sm"
          />
          <p className="text-xs text-muted-foreground">
            {provider === 'openrouter'
              ? 'OpenRouter model ID, e.g. "openai/gpt-4o-mini" or "x-ai/grok-4.1-fast"'
              : 'Gemini model ID, e.g. "gemini-2.5-pro" or "gemini-2.0-flash"'}
          </p>
        </div>

        {/* API Key */}
        <div className="space-y-1.5">
          <Label htmlFor="gate-api-key">API Key</Label>
          <div className="relative flex items-center">
            <Input
              id="gate-api-key"
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={settings?.has_api_key ? '••••••  (key already set — enter new to replace)' : 'Enter API key'}
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
            Stored encrypted at rest. Leave blank to keep the existing key.
          </p>
        </div>

        {/* Save / Test buttons */}
        <div className="flex items-center gap-3 pt-1">
          <Button onClick={handleSave} disabled={saving} size="sm">
            {saving ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={handleTest}
            disabled={testState === 'running' || !isConfigured}
            title={!isConfigured ? 'Save an API key before testing' : undefined}
          >
            {testState === 'running' ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <FlaskConical className="h-4 w-4 mr-2" />
            )}
            Test Model
          </Button>

          {saved && (
            <Badge variant="outline" className="gap-1 text-green-600 border-green-600">
              <CheckCircle2 className="h-3 w-3" />
              Saved
            </Badge>
          )}
        </div>

        {/* Save error */}
        {saveError && (
          <Alert variant="destructive">
            <AlertDescription>{saveError}</AlertDescription>
          </Alert>
        )}

        {/* Test result */}
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
                {testState === 'passed' ? 'Model test passed' : 'Model test failed'}
              </div>
              {testResult.reasoning && (
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {testResult.reasoning}
                </p>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
