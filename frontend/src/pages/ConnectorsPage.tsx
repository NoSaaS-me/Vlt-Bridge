import { useState, useEffect, useCallback, useMemo } from 'react';
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
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  listConnectors,
  getConnectorConfig,
  saveConnectorConfig,
  revokeOAuth,
  type ConnectorInfo,
} from '@/services/connectors';
import {
  getComposioStatus,
  listApps,
  connectApp,
  disconnectApp,
  type ComposioApp,
} from '@/services/composio-hub';

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

// ── App name overrides ───────────────────────────────────────────────────────
// Composio returns lowercase/slugified names; map known apps to proper brand names.
const BRAND_NAMES: Record<string, string> = {
  gmail: 'Gmail', github: 'GitHub', gitlab: 'GitLab', bitbucket: 'Bitbucket',
  notion: 'Notion', slack: 'Slack', discord: 'Discord', telegram: 'Telegram',
  whatsapp: 'WhatsApp', linkedin: 'LinkedIn', twitter: 'Twitter / X',
  youtube: 'YouTube', instagram: 'Instagram', reddit: 'Reddit', tiktok: 'TikTok',
  googlecalendar: 'Google Calendar', googledrive: 'Google Drive',
  googlesheets: 'Google Sheets', googledocs: 'Google Docs',
  googlemeet: 'Google Meet', googletasks: 'Google Tasks',
  googlemaps: 'Google Maps', googlebigquery: 'Google BigQuery',
  googleanalytics: 'Google Analytics', googleads: 'Google Ads',
  googleforms: 'Google Forms', googlechat: 'Google Chat',
  hubspot: 'HubSpot', salesforce: 'Salesforce', pipedrive: 'Pipedrive',
  zohocrm: 'Zoho CRM', freshsales: 'Freshsales', closecrm: 'Close CRM',
  jira: 'Jira', confluence: 'Confluence', trello: 'Trello', asana: 'Asana',
  linear: 'Linear', clickup: 'ClickUp', monday: 'Monday.com', basecamp: 'Basecamp',
  airtable: 'Airtable', webflow: 'Webflow', figma: 'Figma', canva: 'Canva',
  dropbox: 'Dropbox', box: 'Box', onedrive: 'OneDrive', sharepoint: 'SharePoint',
  stripe: 'Stripe', paypal: 'PayPal', quickbooks: 'QuickBooks', xero: 'Xero',
  zoom: 'Zoom', calendly: 'Calendly', cal: 'Cal.com', loom: 'Loom',
  twilio: 'Twilio', sendgrid: 'SendGrid', mailchimp: 'Mailchimp',
  mailgun: 'Mailgun', postmark: 'Postmark', resend: 'Resend',
  intercom: 'Intercom', zendesk: 'Zendesk', freshdesk: 'Freshdesk',
  shopify: 'Shopify', woocommerce: 'WooCommerce', amazon: 'Amazon',
  aws: 'AWS', gcp: 'Google Cloud', azure: 'Azure', vercel: 'Vercel',
  heroku: 'Heroku', render: 'Render', netlify: 'Netlify',
  mongodb: 'MongoDB', supabase: 'Supabase', firebase: 'Firebase',
  postgres: 'PostgreSQL', mysql: 'MySQL', redis: 'Redis',
  openai: 'OpenAI', anthropic: 'Anthropic', cohere: 'Cohere',
  composio: 'Composio', zapier: 'Zapier', make: 'Make', n8n: 'n8n',
  notion_db: 'Notion DB', github_actions: 'GitHub Actions',
};

function formatAppName(app: ComposioApp): string {
  const normalized = app.name.toLowerCase().replace(/[-_\s]/g, '');
  if (BRAND_NAMES[normalized]) return BRAND_NAMES[normalized];
  if (BRAND_NAMES[app.name.toLowerCase()]) return BRAND_NAMES[app.name.toLowerCase()];
  // Title-case the display_name or name as fallback
  const raw = app.display_name || app.name;
  return raw.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function truncateDescription(description: string): string {
  if (!description) return '';
  // Cut at first sentence boundary
  const sentenceEnd = description.search(/[.!?](\s|$)/);
  const first = sentenceEnd >= 0 ? description.slice(0, sentenceEnd + 1) : description;
  return first.length > 100 ? first.slice(0, 97) + '…' : first;
}

// ── Category normalization ────────────────────────────────────────────────────
// Composio categories are very granular and AI-subcategory-heavy.
// Map them to a small set of human-readable buckets.
const CATEGORY_RULES: Array<{ match: RegExp | string; bucket: string }> = [
  // Communication
  { match: /^(email|messaging|chat|communication|sms|phone|video.?call|voip)/i, bucket: 'Communication' },
  { match: 'social media', bucket: 'Social Media' },
  // Productivity
  { match: /^(productivity|notes?|task|calendar|scheduling|time.?tracking|to.?do)/i, bucket: 'Productivity' },
  { match: /^project.?management/i, bucket: 'Productivity' },
  // Developer Tools
  { match: /^(developer|devops|code|version.?control|ci.?cd|monitoring|logging|testing|api)/i, bucket: 'Developer Tools' },
  // Data & Analytics
  { match: /^(data|analytics|bi|reporting|database|spreadsheet|etl)/i, bucket: 'Data & Analytics' },
  // Sales & CRM
  { match: /^(crm|sales|lead|revenue)/i, bucket: 'Sales & CRM' },
  // Marketing
  { match: /^(marketing|ads?|advertising|seo|email.?marketing|growth)/i, bucket: 'Marketing' },
  // Finance
  { match: /^(finance|accounting|payments?|invoic|billing|payroll|tax|bookkeeping)/i, bucket: 'Finance' },
  // Files & Storage
  { match: /^(file|storage|document|cloud.?storage)/i, bucket: 'Files & Storage' },
  // E-commerce
  { match: /^(e.?commerce|shopping|retail|inventory)/i, bucket: 'E-commerce' },
  // HR
  { match: /^(hr|human.?resources|recruiting|hiring|payroll|employee)/i, bucket: 'HR & Recruiting' },
  // Support
  { match: /^(support|customer.?service|helpdesk|ticketing)/i, bucket: 'Customer Support' },
  // Design
  { match: /^(design|creative|media|video|audio|photo)/i, bucket: 'Design & Media' },
  // Automation
  { match: /^(automation|workflow|integration|webhook|trigger)/i, bucket: 'Automation' },
  // Catch-all: any "ai *" subcategory that didn't match above → AI & Automation
  { match: /^ai\b/i, bucket: 'AI & Automation' },
];

function normalizeCategory(raw: string): string {
  for (const rule of CATEGORY_RULES) {
    if (rule.match instanceof RegExp ? rule.match.test(raw) : raw.toLowerCase() === rule.match) {
      return rule.bucket;
    }
  }
  // Title-case anything that fell through
  return raw.replace(/\b\w/g, (c) => c.toUpperCase());
}

// Returns the primary normalized bucket for an app (first matching category)
function appBucket(app: ComposioApp): string {
  for (const cat of app.categories) {
    const bucket = normalizeCategory(cat);
    if (bucket) return bucket;
  }
  return 'Other';
}

// ── Deterministic color avatar for apps without logos ────────────────────────
const AVATAR_COLORS = [
  'bg-blue-500', 'bg-violet-500', 'bg-emerald-500', 'bg-orange-500',
  'bg-pink-500', 'bg-cyan-500', 'bg-amber-500', 'bg-rose-500',
  'bg-indigo-500', 'bg-teal-500',
];

function appAvatarColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return AVATAR_COLORS[h % AVATAR_COLORS.length];
}

function AppCard({
  app,
  connecting,
  disconnecting,
  onConnect,
  onDisconnect,
}: {
  app: ComposioApp;
  connecting: string | null;
  disconnecting: string | null;
  onConnect: (app: ComposioApp) => void;
  onDisconnect: (app: ComposioApp) => void;
}) {
  const displayName = formatAppName(app);
  const initial = displayName[0].toUpperCase();
  const avatarColor = appAvatarColor(app.name);
  const description = truncateDescription(app.description);
  const bucket = appBucket(app);

  return (
    <Card className={`flex flex-col ${app.connected ? 'ring-1 ring-green-400' : ''}`}>
      <CardHeader className="pb-2 flex-1">
        <div className="flex items-start gap-3">
          <div className={`shrink-0 w-9 h-9 rounded-lg ${avatarColor} flex items-center justify-center text-white font-bold text-sm`}>
            {initial}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-1">
              <CardTitle className="text-sm leading-tight truncate">{displayName}</CardTitle>
              {app.connected && (
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-green-500 mt-0.5" />
              )}
            </div>
            <CardDescription className="text-xs mt-0.5">
              {description || <span className="italic text-muted-foreground/60">{bucket}</span>}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {app.connected ? (
          <Button
            variant="outline"
            size="sm"
            className="w-full text-red-600 border-red-300 hover:bg-red-50 text-xs h-7"
            onClick={() => onDisconnect(app)}
            disabled={disconnecting === app.name}
          >
            {disconnecting === app.name ? 'Disconnecting…' : 'Disconnect'}
          </Button>
        ) : (
          <Button
            variant="outline"
            size="sm"
            className="w-full text-xs h-7"
            onClick={() => onConnect(app)}
            disabled={connecting === app.name}
          >
            {connecting === app.name ? 'Opening…' : 'Connect'}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

function HubTab() {
  const [apps, setApps] = useState<ComposioApp[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [connecting, setConnecting] = useState<string | null>(null);
  const [disconnecting, setDisconnecting] = useState<string | null>(null);
  const [configured, setConfigured] = useState<boolean | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [statusRes, appsRes] = await Promise.all([
        getComposioStatus(),
        listApps().catch(() => ({ apps: [], total: 0 })),
      ]);
      setConfigured(statusRes.configured);
      setApps(appsRes.apps);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load');
      setConfigured(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleConnect = async (app: ComposioApp) => {
    setConnecting(app.name);
    try {
      const res = await connectApp(app.name);
      window.open(res.redirect_url, '_blank', 'noopener,noreferrer');
      setTimeout(load, 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connect failed');
    } finally {
      setConnecting(null);
    }
  };

  const handleDisconnect = async (app: ComposioApp) => {
    setDisconnecting(app.name);
    try {
      await disconnectApp(app.name);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Disconnect failed');
    } finally {
      setDisconnecting(null);
    }
  };

  // Build normalized category list with counts
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const app of apps) {
      const bucket = appBucket(app);
      counts[bucket] = (counts[bucket] ?? 0) + 1;
    }
    return counts;
  }, [apps]);

  const categories = useMemo(
    () => Object.keys(categoryCounts).sort(),
    [categoryCounts]
  );

  const connectedCount = apps.filter((a) => a.connected).length;

  // Filter by search + active category
  const searchLower = search.toLowerCase();
  const filtered = useMemo(() => {
    return apps.filter((a) => {
      const displayName = formatAppName(a).toLowerCase();
      const matchesSearch =
        !search ||
        a.name.toLowerCase().includes(searchLower) ||
        displayName.includes(searchLower) ||
        a.categories.some((c) => c.toLowerCase().includes(searchLower)) ||
        appBucket(a).toLowerCase().includes(searchLower);

      const matchesCategory =
        activeCategory === 'all' ||
        (activeCategory === 'connected' ? a.connected : appBucket(a) === activeCategory);

      return matchesSearch && matchesCategory;
    });
  }, [apps, search, searchLower, activeCategory]);

  // When showing "all", pin connected apps to the top
  const { connectedApps, otherApps } = useMemo(() => {
    if (activeCategory !== 'all') return { connectedApps: [], otherApps: filtered };
    return {
      connectedApps: filtered.filter((a) => a.connected),
      otherApps: filtered.filter((a) => !a.connected),
    };
  }, [filtered, activeCategory]);

  if (!configured && !loading) {
    return (
      <div className="p-6 text-center">
        <p className="text-muted-foreground text-sm mb-2">
          Composio is not configured. Set{' '}
          <code className="font-mono bg-muted px-1 rounded">COMPOSIO_API_KEY</code>{' '}
          in your backend environment.
        </p>
        <a
          href="https://app.composio.dev/settings"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-500 hover:underline"
        >
          Get your API key →
        </a>
      </div>
    );
  }

  return (
    <div className="flex gap-4">
      {/* Category sidebar */}
      <div className="w-44 shrink-0">
        <nav className="space-y-0.5 sticky top-0">
          {[
            { key: 'all', label: 'All Apps', count: apps.length },
            { key: 'connected', label: 'Connected', count: connectedCount },
          ].map(({ key, label, count }) => (
            <button
              key={key}
              onClick={() => setActiveCategory(key)}
              className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-md text-sm transition-colors ${
                activeCategory === key
                  ? 'bg-accent text-accent-foreground font-medium'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent/50'
              }`}
            >
              <span className="truncate">{label}</span>
              <span className={`text-xs tabular-nums ml-1 ${activeCategory === key ? '' : 'text-muted-foreground'}`}>
                {count}
              </span>
            </button>
          ))}

          {categories.length > 0 && (
            <div className="pt-2 pb-1">
              <p className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Categories
              </p>
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setActiveCategory(cat)}
                  className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-md text-sm transition-colors ${
                    activeCategory === cat
                      ? 'bg-accent text-accent-foreground font-medium'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent/50'
                  }`}
                >
                  <span className="truncate">{cat}</span>
                  <span className="text-xs tabular-nums ml-1 text-muted-foreground">
                    {categoryCounts[cat]}
                  </span>
                </button>
              ))}
            </div>
          )}
        </nav>
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0 space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="flex items-center gap-2">
          <Input
            placeholder="Search apps…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="max-w-xs"
          />
          <span className="text-xs text-muted-foreground">
            {filtered.length} app{filtered.length !== 1 ? 's' : ''}
          </span>
        </div>

        {loading ? (
          <p className="text-sm text-muted-foreground py-8 text-center">Loading integration catalog…</p>
        ) : filtered.length === 0 ? (
          <p className="text-center text-sm text-muted-foreground py-8">
            No apps match{search ? ` "${search}"` : ' this filter'}
          </p>
        ) : (
          <div className="space-y-6">
            {connectedApps.length > 0 && (
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Connected
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
                  {connectedApps.map((app) => (
                    <AppCard
                      key={app.name}
                      app={app}
                      connecting={connecting}
                      disconnecting={disconnecting}
                      onConnect={handleConnect}
                      onDisconnect={handleDisconnect}
                    />
                  ))}
                </div>
              </div>
            )}

            {otherApps.length > 0 && (
              <div>
                {connectedApps.length > 0 && (
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                    {activeCategory === 'all' ? 'All Integrations' : activeCategory}
                  </p>
                )}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
                  {otherApps.map((app) => (
                    <AppCard
                      key={app.name}
                      app={app}
                      connecting={connecting}
                      disconnecting={disconnecting}
                      onConnect={handleConnect}
                      onDisconnect={handleDisconnect}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
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
    <div className="p-6 max-w-5xl mx-auto space-y-6">
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

      <Tabs defaultValue="native">
        <TabsList>
          <TabsTrigger value="native">Native</TabsTrigger>
          <TabsTrigger value="hub">Integration Hub</TabsTrigger>
        </TabsList>

        <TabsContent value="native" className="mt-4">
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
        </TabsContent>

        <TabsContent value="hub" className="mt-4">
          <HubTab />
        </TabsContent>
      </Tabs>

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
