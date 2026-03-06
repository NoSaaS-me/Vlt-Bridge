import { useState, useEffect, useCallback } from 'react';
import { Settings2, Plug, CheckCircle2, AlertCircle, Link, Unlink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from '@/components/ui/dialog';
import {
  listConnectors,
  getConnectorConfig,
  saveConnectorConfig,
  revokeOAuth,
  type ConnectorInfo,
} from '@/services/connectors';

function ConnectorSettingsDialog({
  connector,
  open,
  onClose,
  onSaved,
}: {
  connector: ConnectorInfo;
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [fields, setFields] = useState<Record<string, string>>({});
  const [enabled, setEnabled] = useState(connector.enabled);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    getConnectorConfig(connector.name)
      .then((cfg) => {
        setEnabled(cfg['__enabled'] === 'true');
        const rest: Record<string, string> = {};
        for (const [k, v] of Object.entries(cfg)) {
          if (!k.startsWith('__')) rest[k] = v;
        }
        setFields(rest);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [open, connector.name]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const config: Record<string, string> = {
        ...fields,
        __enabled: enabled ? 'true' : 'false',
      };
      await saveConnectorConfig(connector.name, config);
      onSaved();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{connector.display_name} Settings</DialogTitle>
        </DialogHeader>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {loading ? (
          <p className="text-sm text-muted-foreground py-4">Loading…</p>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Enable connector</Label>
              <Switch checked={enabled} onCheckedChange={setEnabled} />
            </div>

            {connector.credential_fields.map((field) => (
              <div key={field.name} className="space-y-1">
                <Label htmlFor={`field-${field.name}`}>{field.label}</Label>
                {field.field_type === 'select' ? (
                  <select
                    id={`field-${field.name}`}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring"
                    value={fields[field.name] ?? field.options?.[0] ?? ''}
                    onChange={(e) => setFields((prev) => ({ ...prev, [field.name]: e.target.value }))}
                  >
                    {(field.options ?? []).map((opt) => (
                      <option key={opt} value={opt}>
                        {opt === 'allow_all' ? 'Allow All' : opt.charAt(0).toUpperCase() + opt.slice(1)}
                      </option>
                    ))}
                  </select>
                ) : (
                  <Input
                    id={`field-${field.name}`}
                    type={field.secret ? 'password' : 'text'}
                    placeholder={
                      field.secret && fields[field.name] === '••••••••'
                        ? 'Already set — enter new value to change'
                        : field.placeholder
                    }
                    value={fields[field.name] === '••••••••' ? '' : (fields[field.name] ?? '')}
                    onChange={(e) =>
                      setFields((prev) => ({ ...prev, [field.name]: e.target.value }))
                    }
                  />
                )}
              </div>
            ))}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving || loading}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function OAuthConnectorCard({
  connector,
  onChanged,
}: {
  connector: ConnectorInfo;
  onChanged: () => void;
}) {
  const [revoking, setRevoking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConnect = () => {
    // Full page redirect to backend OAuth authorize endpoint
    window.location.href = `/api/connectors/${connector.name}/oauth/authorize`;
  };

  const handleDisconnect = async () => {
    setRevoking(true);
    setError(null);
    try {
      await revokeOAuth(connector.name);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Disconnect failed');
    } finally {
      setRevoking(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-base">{connector.display_name}</CardTitle>
            <CardDescription className="text-xs">{connector.description}</CardDescription>
          </div>
          <div className="flex gap-2 flex-shrink-0 ml-2">
            {connector.configured ? (
              <Button
                variant="outline"
                size="sm"
                onClick={handleDisconnect}
                disabled={revoking}
                className="text-red-600 border-red-300 hover:bg-red-50"
              >
                <Unlink className="h-4 w-4 mr-1" />
                {revoking ? 'Disconnecting…' : 'Disconnect'}
              </Button>
            ) : (
              <Button variant="default" size="sm" onClick={handleConnect}>
                <Link className="h-4 w-4 mr-1" />
                Connect
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {error && (
          <p className="text-xs text-red-500 mb-2">{error}</p>
        )}
        <div className="flex items-center gap-2 flex-wrap">
          {connector.configured ? (
            <Badge variant="default" className="bg-green-600 text-xs gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Connected
            </Badge>
          ) : (
            <Badge variant="secondary" className="text-xs">
              Not connected
            </Badge>
          )}
          <span className="text-xs text-muted-foreground">
            {connector.actions.length} action{connector.actions.length !== 1 ? 's' : ''}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export function ConnectorsPage() {
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [settingsFor, setSettingsFor] = useState<ConnectorInfo | null>(null);
  const [oauthMessage, setOauthMessage] = useState<string | null>(null);

  // Handle OAuth2 callback redirects: ?connected=<name> or ?oauth_error=<msg>
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const connected = params.get('connected');
    const oauthError = params.get('oauth_error');
    if (connected) {
      setOauthMessage(`Successfully connected ${connected}`);
      window.history.replaceState({}, '', window.location.pathname);
    } else if (oauthError) {
      setError(`OAuth2 error: ${oauthError}`);
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setConnectors(await listConnectors());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load connectors');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Plug className="h-6 w-6 text-muted-foreground" />
        <div>
          <h1 className="text-2xl font-semibold">Connectors</h1>
          <p className="text-sm text-muted-foreground">
            Connect external services that AI agents can use.
          </p>
        </div>
      </div>

      {oauthMessage && (
        <Alert>
          <CheckCircle2 className="h-4 w-4" />
          <AlertDescription>{oauthMessage}</AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading connectors…</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {connectors.map((connector) =>
            connector.auth_type === 'oauth2' ? (
              <OAuthConnectorCard
                key={connector.name}
                connector={connector}
                onChanged={load}
              />
            ) : (
              <Card key={connector.name}>
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-base">{connector.display_name}</CardTitle>
                      <CardDescription className="text-xs">{connector.description}</CardDescription>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-shrink-0 ml-2"
                      onClick={() => setSettingsFor(connector)}
                    >
                      <Settings2 className="h-4 w-4 mr-1" />
                      Settings
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    {connector.enabled && connector.configured ? (
                      <Badge variant="default" className="bg-green-600 text-xs gap-1">
                        <CheckCircle2 className="h-3 w-3" />
                        Enabled
                      </Badge>
                    ) : connector.enabled && !connector.configured ? (
                      <Badge variant="outline" className="text-amber-600 border-amber-600 text-xs gap-1">
                        <AlertCircle className="h-3 w-3" />
                        Needs config
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="text-xs">
                        Disabled
                      </Badge>
                    )}
                    <span className="text-xs text-muted-foreground">
                      {connector.actions.length} action{connector.actions.length !== 1 ? 's' : ''}
                    </span>
                  </div>
                </CardContent>
              </Card>
            )
          )}
        </div>
      )}

      {settingsFor && (
        <ConnectorSettingsDialog
          connector={settingsFor}
          open={true}
          onClose={() => setSettingsFor(null)}
          onSaved={load}
        />
      )}
    </div>
  );
}
