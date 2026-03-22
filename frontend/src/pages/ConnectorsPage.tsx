import { useState, useEffect, useCallback, useMemo } from 'react';
import { Settings2, Plug, CheckCircle2, AlertCircle, Link, Unlink, ChevronDown, ChevronUp, Zap, Plus } from 'lucide-react';
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
  listAppActions,
  getAuthInfo,
  getComposioConfig,
  saveComposioConfig,
  type ComposioApp,
  type ComposioAction,
  type AppAuthInfo,
  type AuthFieldInfo,
} from '@/services/composio-hub';

// ── Action permission types ───────────────────────────────────────────────────

type ActionPermission = 'off' | 'ask' | 'allow';

function actionConfigKey(actionName: string): string {
  return `__action_${actionName}`;
}

function getActionPermission(
  config: Record<string, string>,
  actionName: string
): ActionPermission {
  const val = config[actionConfigKey(actionName)];
  if (val === 'off' || val === 'ask' || val === 'allow') return val;
  return 'allow'; // default
}

// ── 3-way toggle ─────────────────────────────────────────────────────────────

function ActionPermissionToggle({
  value,
  onChange,
  disabled,
}: {
  value: ActionPermission;
  onChange: (v: ActionPermission) => void;
  disabled?: boolean;
}) {
  const options: { label: string; value: ActionPermission; active: string; inactive: string }[] = [
    {
      label: 'Off',
      value: 'off',
      active: 'bg-muted text-muted-foreground border-border font-medium',
      inactive: 'text-muted-foreground/50 border-transparent hover:border-border/50',
    },
    {
      label: 'Ask',
      value: 'ask',
      active: 'bg-amber-900/30 text-amber-400 border-amber-700 font-medium',
      inactive: 'text-muted-foreground/50 border-transparent hover:border-border/50',
    },
    {
      label: 'Allow',
      value: 'allow',
      active: 'bg-green-900/30 text-green-400 border-green-700 font-medium',
      inactive: 'text-muted-foreground/50 border-transparent hover:border-border/50',
    },
  ];

  return (
    <div className="flex rounded-md shrink-0" style={{ width: 120 }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={(e) => { e.stopPropagation(); if (!disabled) onChange(opt.value); }}
          disabled={disabled}
          className={`flex-1 text-[10px] py-0.5 border rounded-sm transition-colors ${value === opt.value ? opt.active : opt.inactive} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

// ── Action list with collapsing ───────────────────────────────────────────────

const ACTIONS_VISIBLE_LIMIT = 4;

interface ActionRowProps {
  action: { name: string; description: string };
  permission: ActionPermission;
  onPermissionChange: (perm: ActionPermission) => void;
  saving: boolean;
}

function ActionRow({ action, permission, onPermissionChange, saving }: ActionRowProps) {
  return (
    <div className="flex items-center gap-2 py-1.5 min-w-0 group/row">
      <div className="flex-1 min-w-0">
        <code className="text-[10px] font-mono text-foreground/80 block truncate" title={action.name}>
          {action.name}
        </code>
        {action.description && (
          <span className="text-[10px] text-muted-foreground/60 block truncate leading-tight" title={action.description}>
            {action.description}
          </span>
        )}
      </div>
      <ActionPermissionToggle
        value={permission}
        onChange={onPermissionChange}
        disabled={saving}
      />
    </div>
  );
}

interface ActionListProps {
  actions: Array<{ name: string; description: string }>;
  config: Record<string, string>;
  onPermissionChange: (actionName: string, perm: ActionPermission) => void;
  savingAction: string | null;
}

function ActionList({ actions, config, onPermissionChange, savingAction }: ActionListProps) {
  const [expanded, setExpanded] = useState(false);

  if (actions.length === 0) return null;

  const visible = expanded ? actions : actions.slice(0, ACTIONS_VISIBLE_LIMIT);
  const hiddenCount = actions.length - ACTIONS_VISIBLE_LIMIT;

  return (
    <div className="mt-2 border-t border-border/40 pt-2 space-y-0">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70 mb-0.5">
        Actions ({actions.length})
      </p>
      {visible.map((action) => (
        <ActionRow
          key={action.name}
          action={action}
          permission={getActionPermission(config, action.name)}
          onPermissionChange={(perm) => onPermissionChange(action.name, perm)}
          saving={savingAction === action.name}
        />
      ))}
      {hiddenCount > 0 && (
        <button
          onClick={(e) => { e.stopPropagation(); setExpanded((prev) => !prev); }}
          className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors mt-1"
        >
          {expanded ? (
            <><ChevronUp className="h-3 w-3" /> Show less</>
          ) : (
            <><ChevronDown className="h-3 w-3" /> Show {hiddenCount} more</>
          )}
        </button>
      )}
    </div>
  );
}

// ── Connector settings dialog ─────────────────────────────────────────────────

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

// ── Native connector card (API key / service) ─────────────────────────────────

function NativeConnectorCard({
  connector,
  onSettingsClick,
  onChanged,
}: {
  connector: ConnectorInfo;
  onSettingsClick: () => void;
  onChanged: () => void;
}) {
  // Per-connector action permission config state
  const [config, setConfig] = useState<Record<string, string>>({});
  const [configLoaded, setConfigLoaded] = useState(false);
  const [savingAction, setSavingAction] = useState<string | null>(null);

  // Load config once when the card mounts
  useEffect(() => {
    getConnectorConfig(connector.name)
      .then((cfg) => {
        setConfig(cfg);
        setConfigLoaded(true);
      })
      .catch(() => setConfigLoaded(true)); // silently proceed if config fetch fails
  }, [connector.name]);

  const handlePermissionChange = useCallback(async (actionName: string, perm: ActionPermission) => {
    const key = actionConfigKey(actionName);
    // Optimistic update
    setConfig((prev) => ({ ...prev, [key]: perm }));
    setSavingAction(actionName);
    try {
      await saveConnectorConfig(connector.name, { [key]: perm });
      onChanged();
    } catch {
      // Roll back on error
      setConfig((prev) => {
        const reverted = { ...prev };
        delete reverted[key];
        return reverted;
      });
    } finally {
      setSavingAction(null);
    }
  }, [connector.name, onChanged]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="space-y-1 min-w-0 flex-1">
            <CardTitle className="text-base">{connector.display_name}</CardTitle>
            <CardDescription className="text-xs">{connector.description}</CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="flex-shrink-0 ml-2"
            onClick={onSettingsClick}
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

        {configLoaded && connector.actions.length > 0 && (
          <ActionList
            actions={connector.actions}
            config={config}
            onPermissionChange={handlePermissionChange}
            savingAction={savingAction}
          />
        )}
      </CardContent>
    </Card>
  );
}

// ── OAuth connector card ──────────────────────────────────────────────────────

function OAuthConnectorCard({
  connector,
  onChanged,
}: {
  connector: ConnectorInfo;
  onChanged: () => void;
}) {
  const [revoking, setRevoking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<Record<string, string>>({});
  const [configLoaded, setConfigLoaded] = useState(false);
  const [savingAction, setSavingAction] = useState<string | null>(null);

  useEffect(() => {
    getConnectorConfig(connector.name)
      .then((cfg) => {
        setConfig(cfg);
        setConfigLoaded(true);
      })
      .catch(() => setConfigLoaded(true));
  }, [connector.name]);

  const handleConnect = () => {
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

  const handlePermissionChange = useCallback(async (actionName: string, perm: ActionPermission) => {
    const key = actionConfigKey(actionName);
    setConfig((prev) => ({ ...prev, [key]: perm }));
    setSavingAction(actionName);
    try {
      await saveConnectorConfig(connector.name, { [key]: perm });
      onChanged();
    } catch {
      setConfig((prev) => {
        const reverted = { ...prev };
        delete reverted[key];
        return reverted;
      });
    } finally {
      setSavingAction(null);
    }
  }, [connector.name, onChanged]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="space-y-1 min-w-0 flex-1">
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

        {configLoaded && connector.actions.length > 0 && (
          <ActionList
            actions={connector.actions}
            config={config}
            onPermissionChange={handlePermissionChange}
            savingAction={savingAction}
          />
        )}
      </CardContent>
    </Card>
  );
}

// ── App name overrides ────────────────────────────────────────────────────────
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
  const raw = app.display_name || app.name;
  return raw.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function truncateDescription(description: string): string {
  if (!description) return '';
  const sentenceEnd = description.search(/[.!?](\s|$)/);
  const first = sentenceEnd >= 0 ? description.slice(0, sentenceEnd + 1) : description;
  return first.length > 100 ? first.slice(0, 97) + '…' : first;
}

// ── Category normalization ────────────────────────────────────────────────────
const CATEGORY_RULES: Array<{ match: RegExp | string; bucket: string }> = [
  { match: /^(email|messaging|chat|communication|sms|phone|video.?call|voip)/i, bucket: 'Communication' },
  { match: 'social media', bucket: 'Social Media' },
  { match: /^(productivity|notes?|task|calendar|scheduling|time.?tracking|to.?do)/i, bucket: 'Productivity' },
  { match: /^project.?management/i, bucket: 'Productivity' },
  { match: /^(developer|devops|code|version.?control|ci.?cd|monitoring|logging|testing|api)/i, bucket: 'Developer Tools' },
  { match: /^(data|analytics|bi|reporting|database|spreadsheet|etl)/i, bucket: 'Data & Analytics' },
  { match: /^(crm|sales|lead|revenue)/i, bucket: 'Sales & CRM' },
  { match: /^(marketing|ads?|advertising|seo|email.?marketing|growth)/i, bucket: 'Marketing' },
  { match: /^(finance|accounting|payments?|invoic|billing|payroll|tax|bookkeeping)/i, bucket: 'Finance' },
  { match: /^(file|storage|document|cloud.?storage)/i, bucket: 'Files & Storage' },
  { match: /^(e.?commerce|shopping|retail|inventory)/i, bucket: 'E-commerce' },
  { match: /^(hr|human.?resources|recruiting|hiring|payroll|employee)/i, bucket: 'HR & Recruiting' },
  { match: /^(support|customer.?service|helpdesk|ticketing)/i, bucket: 'Customer Support' },
  { match: /^(design|creative|media|video|audio|photo)/i, bucket: 'Design & Media' },
  { match: /^(automation|workflow|integration|webhook|trigger)/i, bucket: 'Automation' },
  { match: /^ai\b/i, bucket: 'AI & Automation' },
];

function normalizeCategory(raw: string): string {
  for (const rule of CATEGORY_RULES) {
    if (rule.match instanceof RegExp ? rule.match.test(raw) : raw.toLowerCase() === rule.match) {
      return rule.bucket;
    }
  }
  return raw.replace(/\b\w/g, (c) => c.toUpperCase());
}

function appBucket(app: ComposioApp): string {
  for (const cat of app.categories) {
    const bucket = normalizeCategory(cat);
    if (bucket) return bucket;
  }
  return 'Other';
}

// ── Deterministic color avatar ────────────────────────────────────────────────
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

// ── Composio app card ─────────────────────────────────────────────────────────

function AppCard({
  app,
  connecting,
  disconnecting,
  onConnect,
  onDisconnect,
  onPreview,
}: {
  app: ComposioApp;
  connecting: string | null;
  disconnecting: string | null;
  onConnect: (app: ComposioApp) => void;
  onDisconnect: (app: ComposioApp) => void;
  onPreview: (app: ComposioApp) => void;
}) {
  const displayName = formatAppName(app);
  const initial = displayName[0].toUpperCase();
  const avatarColor = appAvatarColor(app.name);
  const description = truncateDescription(app.description);
  const bucket = appBucket(app);

  // Action permissions state for connected composio apps
  const [actions, setActions] = useState<ComposioAction[]>([]);
  const [actionsLoaded, setActionsLoaded] = useState(false);
  const [composioConfig, setComposioConfig] = useState<Record<string, string>>({});
  const [savingAction, setSavingAction] = useState<string | null>(null);

  useEffect(() => {
    if (!app.connected) return;
    Promise.all([
      listAppActions(app.name).then((res) => setActions(res.actions)),
      getComposioConfig(app.name).then((res) => setComposioConfig(res.config)),
    ])
      .catch(() => {})
      .finally(() => setActionsLoaded(true));
  }, [app.name, app.connected]);

  const handlePermissionChange = useCallback(async (actionName: string, perm: ActionPermission) => {
    const key = actionConfigKey(actionName);
    const prev = composioConfig[key];
    // Optimistic update
    setComposioConfig((c) => ({ ...c, [key]: perm }));
    setSavingAction(actionName);
    try {
      await saveComposioConfig(app.name, { [key]: perm });
    } catch {
      // Roll back on error
      setComposioConfig((c) => prev ? { ...c, [key]: prev } : (() => { const r = { ...c }; delete r[key]; return r; })());
    } finally {
      setSavingAction(null);
    }
  }, [app.name, composioConfig]);

  const composioActions = useMemo(
    () => actions.map((a) => ({ name: a.name, description: a.description })),
    [actions]
  );

  return (
    <Card
      className={`flex flex-col cursor-pointer hover:border-border transition-colors ${app.connected ? 'ring-1 ring-emerald-500/30 border-emerald-500/20' : ''}`}
      onClick={() => onPreview(app)}
    >
      <CardHeader className="pb-2 flex-1">
        <div className="flex items-start gap-3">
          <div className={`shrink-0 w-9 h-9 rounded-lg ${avatarColor} flex items-center justify-center text-white font-bold text-sm`}>
            {initial}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-1">
              <CardTitle className="text-sm leading-tight truncate">{displayName}</CardTitle>
              {app.connected && (
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-400 mt-0.5" />
              )}
            </div>
            <CardDescription className="text-xs mt-0.5 line-clamp-2">
              {description || <span className="italic text-muted-foreground/60">{bucket}</span>}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-2">
        {app.connected ? (
          <div className="flex gap-1.5">
            <Button
              variant="outline"
              size="sm"
              className="flex-1 text-xs h-7 text-red-400 border-red-500/30 hover:bg-red-500/10 hover:text-red-300"
              onClick={(e) => { e.stopPropagation(); onDisconnect(app); }}
              disabled={disconnecting === app.name}
            >
              {disconnecting === app.name ? (
                'Disconnecting…'
              ) : (
                <><Unlink className="h-3 w-3 mr-1.5" />Disconnect</>
              )}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="text-xs h-7 px-2"
              onClick={(e) => { e.stopPropagation(); onConnect(app); }}
              title="Add another connection"
            >
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        ) : (
          <Button
            variant="outline"
            size="sm"
            className="w-full text-xs h-7"
            onClick={(e) => { e.stopPropagation(); onConnect(app); }}
            disabled={connecting === app.name}
          >
            {connecting === app.name ? (
              'Opening…'
            ) : (
              <><Link className="h-3 w-3 mr-1.5" />Connect</>
            )}
          </Button>
        )}

        {app.connected && actionsLoaded && composioActions.length > 0 && (
          <ActionList
            actions={composioActions}
            config={composioConfig}
            onPermissionChange={handlePermissionChange}
            savingAction={savingAction}
          />
        )}
      </CardContent>
    </Card>
  );
}

// ── App actions preview dialog ────────────────────────────────────────────────

function AppActionsPreviewDialog({
  app,
  open,
  onClose,
  onConnect,
  connecting,
}: {
  app: ComposioApp | null;
  open: boolean;
  onClose: () => void;
  onConnect: (app: ComposioApp) => void;
  connecting: string | null;
}) {
  const [actions, setActions] = useState<ComposioAction[]>([]);
  const [total, setTotal] = useState(0);
  const [loadingActions, setLoadingActions] = useState(false);

  useEffect(() => {
    if (!open || !app) return;
    setActions([]);
    setTotal(0);
    setLoadingActions(true);
    listAppActions(app.name)
      .then((res) => {
        setActions(res.actions);
        setTotal(res.total);
      })
      .catch(() => {})
      .finally(() => setLoadingActions(false));
  }, [open, app]);

  if (!app) return null;

  const displayName = formatAppName(app);
  const initial = displayName[0].toUpperCase();
  const avatarColor = appAvatarColor(app.name);
  const bucket = appBucket(app);

  const actionCount = total > 0 ? total : actions.length;

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-lg max-h-[85vh] flex flex-col gap-0 p-0">
        {/* Fixed header */}
        <div className="px-6 pt-6 pb-3 shrink-0">
          <DialogHeader>
            <div className="flex items-center gap-3">
              <div className={`shrink-0 w-10 h-10 rounded-lg ${avatarColor} flex items-center justify-center text-white font-bold text-sm`}>
                {initial}
              </div>
              <div className="min-w-0 flex-1">
                <DialogTitle className="text-base leading-tight">{displayName}</DialogTitle>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="secondary" className="text-[10px]">{bucket}</Badge>
                  {app.connected && (
                    <Badge variant="default" className="bg-emerald-600 text-[10px] gap-1">
                      <CheckCircle2 className="h-3 w-3" />
                      Connected
                    </Badge>
                  )}
                </div>
              </div>
            </div>
          </DialogHeader>

          {app.description && (
            <p className="text-sm text-muted-foreground leading-relaxed mt-3 line-clamp-3">{app.description}</p>
          )}

          <p className="text-xs text-muted-foreground mt-3 pt-3 border-t border-border">
            {actionCount} action{actionCount !== 1 ? 's' : ''} available
          </p>
        </div>

        {/* Scrollable action list */}
        <div className="flex-1 min-h-0 overflow-y-auto px-6">
          {loadingActions ? (
            <p className="text-sm text-muted-foreground py-4 text-center">Loading actions…</p>
          ) : actions.length > 0 ? (
            <div className="divide-y divide-border/40">
              {actions.map((action) => (
                <div key={action.name} className="py-2.5">
                  <p className="font-mono text-xs text-foreground/90 truncate" title={action.name}>
                    {action.display_name || action.name}
                  </p>
                  {action.description && (
                    <p className="text-[11px] text-muted-foreground/70 mt-0.5 leading-snug line-clamp-2" title={action.description}>
                      {action.description}
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground py-2 text-center">No actions found.</p>
          )}
        </div>

        {/* Fixed footer */}
        <div className="px-6 py-4 border-t border-border shrink-0 flex items-center justify-between">
          {app.connected ? (
            <div className="flex items-center gap-1.5 text-sm text-emerald-400">
              <CheckCircle2 className="h-4 w-4" />
              <span>Connected</span>
            </div>
          ) : (
            <Button
              size="sm"
              onClick={() => { onConnect(app); onClose(); }}
              disabled={connecting === app.name}
            >
              <Link className="h-3.5 w-3.5 mr-1.5" />
              {connecting === app.name ? 'Opening…' : 'Connect'}
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ── Connect dialog (adaptive auth form) ──────────────────────────────────────

function ComposioConnectDialog({
  app,
  open,
  onClose,
  onConnected,
}: {
  app: ComposioApp | null;
  open: boolean;
  onClose: () => void;
  onConnected: () => void;
}) {
  const [authInfo, setAuthInfo] = useState<AppAuthInfo | null>(null);
  const [loadingAuth, setLoadingAuth] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [label, setLabel] = useState('');
  const [fieldValues, setFieldValues] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!open || !app) return;
    setAuthInfo(null);
    setError(null);
    setLabel('');
    setFieldValues({});
    setLoadingAuth(true);
    getAuthInfo(app.name)
      .then(setAuthInfo)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load auth info'))
      .finally(() => setLoadingAuth(false));
  }, [open, app]);

  if (!app) return null;

  // Determine which fields to show based on auth info
  const scheme = authInfo?.auth_schemes?.[0];
  const integrationFields: AuthFieldInfo[] = scheme?.integration_fields ?? [];
  const userFields: AuthFieldInfo[] = scheme?.user_fields ?? [];
  const needsCredentials = authInfo && !authInfo.has_managed_auth && integrationFields.length > 0;
  const needsUserParams = userFields.length > 0;
  const allFields = [...integrationFields, ...userFields];

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      // Build auth_config from integration fields (client_id, client_secret)
      const authConfig: Record<string, string> = {};
      for (const f of integrationFields) {
        if (fieldValues[f.name]) authConfig[f.name] = fieldValues[f.name];
      }
      // Build connected_account_params from user fields (api_key, etc.)
      const accountParams: Record<string, string> = {};
      for (const f of userFields) {
        if (fieldValues[f.name]) accountParams[f.name] = fieldValues[f.name];
      }

      const result = await connectApp(app.name, {
        label: label.trim(),
        auth_mode: scheme?.auth_mode,
        auth_config: Object.keys(authConfig).length > 0 ? authConfig : undefined,
        connected_account_params: Object.keys(accountParams).length > 0 ? accountParams : undefined,
      });

      if (result.redirect_url) {
        window.open(result.redirect_url, '_blank', 'noopener,noreferrer');
      }
      onClose();
      // Reload after a delay to let OAuth complete
      setTimeout(onConnected, result.redirect_url ? 3000 : 500);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connection failed');
    } finally {
      setSubmitting(false);
    }
  };

  const displayName = formatAppName(app);

  // Check required fields are filled
  const requiredMissing = allFields.some(
    (f) => f.required && !fieldValues[f.name]?.trim()
  );

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Connect {displayName}</DialogTitle>
        </DialogHeader>

        {loadingAuth ? (
          <p className="text-sm text-muted-foreground py-4 text-center">Checking auth requirements…</p>
        ) : error ? (
          <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>
        ) : authInfo ? (
          <div className="space-y-4">
            {authInfo.has_managed_auth && !needsUserParams && (
              <p className="text-sm text-muted-foreground">
                This app uses Composio-managed OAuth. Click Connect to authorize.
              </p>
            )}

            {needsCredentials && (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  This app requires your own OAuth credentials.
                </p>
                {integrationFields.map((f) => (
                  <div key={f.name} className="space-y-1">
                    <Label className="text-xs">
                      {f.display_name}
                      {f.required && <span className="text-red-400 ml-0.5">*</span>}
                    </Label>
                    {f.description && (
                      <p className="text-[11px] text-muted-foreground/70">{f.description}</p>
                    )}
                    <Input
                      type={f.name.includes('secret') || f.name.includes('password') ? 'password' : 'text'}
                      placeholder={f.display_name}
                      value={fieldValues[f.name] ?? ''}
                      onChange={(e) => setFieldValues((prev) => ({ ...prev, [f.name]: e.target.value }))}
                    />
                  </div>
                ))}
              </div>
            )}

            {needsUserParams && (
              <div className="space-y-3">
                {!needsCredentials && (
                  <p className="text-sm text-muted-foreground">
                    Enter your credentials for this app.
                  </p>
                )}
                {userFields.map((f) => (
                  <div key={f.name} className="space-y-1">
                    <Label className="text-xs">
                      {f.display_name}
                      {f.required && <span className="text-red-400 ml-0.5">*</span>}
                    </Label>
                    {f.description && (
                      <p className="text-[11px] text-muted-foreground/70">{f.description}</p>
                    )}
                    <Input
                      type={f.name.includes('key') || f.name.includes('secret') || f.name.includes('token') || f.name.includes('password') ? 'password' : 'text'}
                      placeholder={f.display_name}
                      value={fieldValues[f.name] ?? ''}
                      onChange={(e) => setFieldValues((prev) => ({ ...prev, [f.name]: e.target.value }))}
                    />
                  </div>
                ))}
              </div>
            )}

            <div className="space-y-1">
              <Label className="text-xs">Label (optional)</Label>
              <Input
                placeholder={`e.g. "Work ${displayName}"`}
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                maxLength={100}
              />
            </div>

            <Badge variant="secondary" className="text-[10px]">
              {authInfo.primary_auth_mode}
            </Badge>
          </div>
        ) : null}

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
          <Button
            size="sm"
            onClick={handleSubmit}
            disabled={submitting || loadingAuth || !authInfo || (needsCredentials && requiredMissing)}
          >
            <Link className="h-3.5 w-3.5 mr-1.5" />
            {submitting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Hub tab ───────────────────────────────────────────────────────────────────

function HubTab() {
  const [apps, setApps] = useState<ComposioApp[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [connecting, setConnecting] = useState<string | null>(null);
  const [disconnecting, setDisconnecting] = useState<string | null>(null);
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [previewApp, setPreviewApp] = useState<ComposioApp | null>(null);
  const [connectDialogApp, setConnectDialogApp] = useState<ComposioApp | null>(null);

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

  const handleConnect = (app: ComposioApp) => {
    setConnectDialogApp(app);
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
        <nav className="space-y-0.5 sticky top-0 max-h-screen overflow-y-auto pb-6">
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

        {/* Composio attribution + search bar */}
        <div className="flex items-center justify-between gap-3">
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
          <a
            href="https://composio.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-center gap-1.5 rounded-full border border-purple-500/30 bg-purple-500/10 px-3 py-1 text-xs font-medium text-purple-400 transition-all hover:border-purple-500/50 hover:bg-purple-500/20 hover:text-purple-300 shrink-0"
          >
            <Zap className="h-3 w-3 text-purple-400 group-hover:text-purple-300 transition-colors" />
            Powered by Composio
          </a>
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
                      onPreview={setPreviewApp}
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
                      onPreview={setPreviewApp}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <AppActionsPreviewDialog
        app={previewApp}
        open={previewApp !== null}
        onClose={() => setPreviewApp(null)}
        onConnect={handleConnect}
        connecting={connecting}
      />

      <ComposioConnectDialog
        app={connectDialogApp}
        open={connectDialogApp !== null}
        onClose={() => setConnectDialogApp(null)}
        onConnected={load}
      />
    </div>
  );
}

// ── ConnectorsPage ────────────────────────────────────────────────────────────

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
                  <NativeConnectorCard
                    key={connector.name}
                    connector={connector}
                    onSettingsClick={() => setSettingsFor(connector)}
                    onChanged={load}
                  />
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
