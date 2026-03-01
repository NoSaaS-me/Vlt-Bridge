/**
 * EntryWizard — 3-step dialog for creating CronbanEntry records.
 *
 * Step 1: Schedule (cron expression / gate column / webhook note) + title + target
 * Step 2: Skill (saved skill OR inline prompt)
 * Step 3: Eval (verifier-only criterion, color picker)
 *
 * Used by CalendarView (type=cron) and KanbanBoard (type=gate).
 */
import { useState, useEffect, useCallback } from 'react';
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Clock,
  Link2,
  Columns3,
  Loader2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  type CronbanEntry,
  type CronbanSkill,
  type KanbanColumn,
  listSkills,
  listColumns,
  createEntry,
} from '@/services/cronban-api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EntryWizardProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId?: string;
  initialType?: 'cron' | 'gate' | 'webhook';
  onCreated: (entry: CronbanEntry) => void;
}

type EntryType = 'cron' | 'gate' | 'webhook';

interface WizardState {
  // Step 1
  entry_type: EntryType;
  title: string;
  cron_expression: string;
  timezone: string;
  kanban_column_id: string;
  gate_check_interval_minutes: number;
  target_cwd: string;
  create_new_session: boolean;
  // Step 2
  skill_mode: 'saved' | 'inline';
  skill_id: string;
  prompt_text: string;
  // Step 3
  eval_text: string;
  color: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIMEZONES = [
  'UTC',
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Los_Angeles',
  'Europe/London',
  'Europe/Paris',
  'Europe/Berlin',
  'Asia/Tokyo',
  'Asia/Shanghai',
  'Australia/Sydney',
];

const CRON_PRESETS = [
  { label: 'Daily 9am', value: '0 9 * * *' },
  { label: 'Weekdays 9am', value: '0 9 * * 1-5' },
  { label: 'Weekly Monday', value: '0 9 * * 1' },
  { label: 'Hourly', value: '0 * * * *' },
  { label: 'Every 30 min', value: '*/30 * * * *' },
];

const COLOR_SWATCHES = [
  { name: 'blue', bg: 'bg-blue-500', ring: 'ring-blue-500' },
  { name: 'emerald', bg: 'bg-emerald-500', ring: 'ring-emerald-500' },
  { name: 'amber', bg: 'bg-amber-500', ring: 'ring-amber-500' },
  { name: 'rose', bg: 'bg-rose-500', ring: 'ring-rose-500' },
  { name: 'purple', bg: 'bg-purple-500', ring: 'ring-purple-500' },
  { name: 'slate', bg: 'bg-slate-500', ring: 'ring-slate-500' },
];

const GATE_INTERVALS = [
  { label: '5 minutes', value: 5 },
  { label: '15 minutes', value: 15 },
  { label: '30 minutes', value: 30 },
  { label: '1 hour', value: 60 },
  { label: '4 hours', value: 240 },
  { label: '24 hours', value: 1440 },
];

const STEPS = ['Schedule', 'Skill', 'Eval'];

// ---------------------------------------------------------------------------
// Step indicators
// ---------------------------------------------------------------------------

function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          className={cn(
            'h-1.5 rounded-full transition-all',
            i < current
              ? 'w-6 bg-primary'
              : i === current
                ? 'w-6 bg-primary'
                : 'w-3 bg-muted',
          )}
        />
      ))}
      <span className="text-xs text-muted-foreground ml-1">
        {current + 1} / {total}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 1: Schedule
// ---------------------------------------------------------------------------

function Step1Schedule({
  state,
  onChange,
  columns,
}: {
  state: WizardState;
  onChange: (patch: Partial<WizardState>) => void;
  columns: KanbanColumn[];
}) {
  return (
    <div className="space-y-4">
      {/* Entry type selector */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted-foreground font-medium">Entry Type</label>
        <div className="flex gap-2">
          {(
            [
              { type: 'cron' as EntryType, icon: Clock, label: 'Cron' },
              { type: 'gate' as EntryType, icon: Columns3, label: 'Gate' },
              { type: 'webhook' as EntryType, icon: Link2, label: 'Webhook' },
            ] as const
          ).map(({ type, icon: Icon, label }) => (
            <button
              key={type}
              onClick={() => onChange({ entry_type: type })}
              className={cn(
                'flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md border text-sm font-medium transition-colors',
                state.entry_type === type
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border hover:bg-muted/50 text-muted-foreground',
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Title (always) */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted-foreground font-medium">
          Title <span className="text-destructive">*</span>
        </label>
        <Input
          value={state.title}
          onChange={(e) => onChange({ title: e.target.value })}
          placeholder="e.g. Daily code review"
          className="h-8 text-sm"
        />
      </div>

      {/* Cron-specific fields */}
      {state.entry_type === 'cron' && (
        <>
          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground font-medium">
              Cron Expression <span className="text-destructive">*</span>
            </label>
            <Input
              value={state.cron_expression}
              onChange={(e) => onChange({ cron_expression: e.target.value })}
              placeholder="0 9 * * 1-5"
              className="h-8 text-sm font-mono"
            />
            {/* Presets */}
            <div className="flex flex-wrap gap-1.5">
              {CRON_PRESETS.map((p) => (
                <button
                  key={p.value}
                  onClick={() => onChange({ cron_expression: p.value })}
                  className={cn(
                    'text-[10px] px-2 py-0.5 rounded border transition-colors',
                    state.cron_expression === p.value
                      ? 'border-primary/60 bg-primary/10 text-primary'
                      : 'border-border hover:bg-muted/50 text-muted-foreground',
                  )}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground font-medium">Timezone</label>
            <Select
              value={state.timezone}
              onValueChange={(v) => onChange({ timezone: v })}
            >
              <SelectTrigger className="h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TIMEZONES.map((tz) => (
                  <SelectItem key={tz} value={tz} className="text-sm">
                    {tz}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </>
      )}

      {/* Gate-specific fields */}
      {state.entry_type === 'gate' && (
        <>
          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground font-medium">Kanban Column</label>
            <Select
              value={state.kanban_column_id}
              onValueChange={(v) => onChange({ kanban_column_id: v })}
            >
              <SelectTrigger className="h-8 text-sm">
                <SelectValue placeholder="Select a column…" />
              </SelectTrigger>
              <SelectContent>
                {columns.map((c) => (
                  <SelectItem key={c.id} value={c.id} className="text-sm">
                    {c.name}
                  </SelectItem>
                ))}
                {columns.length === 0 && (
                  <SelectItem value="__none" disabled className="text-sm text-muted-foreground">
                    No columns found
                  </SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground font-medium">Check Interval</label>
            <Select
              value={String(state.gate_check_interval_minutes)}
              onValueChange={(v) => onChange({ gate_check_interval_minutes: Number(v) })}
            >
              <SelectTrigger className="h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {GATE_INTERVALS.map((gi) => (
                  <SelectItem key={gi.value} value={String(gi.value)} className="text-sm">
                    {gi.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </>
      )}

      {/* Webhook note */}
      {state.entry_type === 'webhook' && (
        <div className="rounded-md border border-blue-500/30 bg-blue-500/5 px-3 py-2.5">
          <p className="text-xs text-blue-400">
            A webhook URL and secret will be shown after the entry is created. Claude will be
            triggered when your external service POSTs to that URL.
          </p>
        </div>
      )}

      {/* Target CWD + new session (always) */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted-foreground font-medium">
          Target Working Directory
        </label>
        <Input
          value={state.target_cwd}
          onChange={(e) => onChange({ target_cwd: e.target.value })}
          placeholder="/home/user/my-project"
          className="h-8 text-sm font-mono"
        />
      </div>

      <label className="flex items-center gap-2 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={state.create_new_session}
          onChange={(e) => onChange({ create_new_session: e.target.checked })}
          className="h-3.5 w-3.5 rounded border-border accent-primary"
        />
        <span className="text-xs text-muted-foreground">Always create a new Claude session</span>
      </label>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 2: Skill
// ---------------------------------------------------------------------------

function Step2Skill({
  state,
  onChange,
  skills,
}: {
  state: WizardState;
  onChange: (patch: Partial<WizardState>) => void;
  skills: CronbanSkill[];
}) {
  const selectedSkill = skills.find((s) => s.id === state.skill_id) ?? null;

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted-foreground font-medium">Prompt Source</label>
        <div className="flex gap-2">
          {(
            [
              { mode: 'saved' as const, label: 'Use a saved Skill' },
              { mode: 'inline' as const, label: 'Write inline prompt' },
            ] as const
          ).map(({ mode, label }) => (
            <button
              key={mode}
              onClick={() => onChange({ skill_mode: mode })}
              className={cn(
                'flex-1 px-3 py-2 rounded-md border text-sm font-medium transition-colors',
                state.skill_mode === mode
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border hover:bg-muted/50 text-muted-foreground',
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {state.skill_mode === 'saved' && (
        <>
          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground font-medium">Skill</label>
            <Select
              value={state.skill_id}
              onValueChange={(v) => onChange({ skill_id: v })}
            >
              <SelectTrigger className="h-8 text-sm">
                <SelectValue placeholder="Select a skill…" />
              </SelectTrigger>
              <SelectContent>
                {skills.map((s) => (
                  <SelectItem key={s.id} value={s.id} className="text-sm">
                    {s.name}
                  </SelectItem>
                ))}
                {skills.length === 0 && (
                  <SelectItem value="__none" disabled className="text-sm text-muted-foreground">
                    No skills found — create one in the Skills tab
                  </SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>

          {selectedSkill && (
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground font-medium flex items-center gap-1.5">
                Preview
                <span className="text-[9px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1 py-0.5 rounded">
                  Claude sees this
                </span>
              </label>
              <pre className="rounded-md border border-border bg-muted/30 p-3 text-xs font-mono text-foreground overflow-y-auto max-h-48 whitespace-pre-wrap leading-relaxed">
                {selectedSkill.prompt_markdown}
              </pre>
            </div>
          )}

          {!selectedSkill && state.skill_id && state.skill_id !== '__none' && (
            <p className="text-xs text-muted-foreground">Loading skill preview…</p>
          )}
        </>
      )}

      {state.skill_mode === 'inline' && (
        <div className="space-y-1.5">
          <label className="text-xs text-muted-foreground font-medium flex items-center gap-1.5">
            Inline Prompt
            <span className="text-[9px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1 py-0.5 rounded">
              Claude sees this
            </span>
          </label>
          <Textarea
            value={state.prompt_text}
            onChange={(e) => onChange({ prompt_text: e.target.value })}
            placeholder="Describe the task in markdown. Claude will receive this as its instructions."
            className="min-h-40 text-sm font-mono resize-none"
          />
          <p className="text-[10px] text-muted-foreground">
            Supports markdown. The eval (hidden from Claude) is set in the next step.
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 3: Eval
// ---------------------------------------------------------------------------

function Step3Eval({
  state,
  onChange,
}: {
  state: WizardState;
  onChange: (patch: Partial<WizardState>) => void;
}) {
  return (
    <div className="space-y-4">
      {/* Warning banner */}
      <div className="rounded-md border border-amber-500/40 bg-amber-500/8 px-3 py-2.5 flex gap-2.5">
        <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
        <div className="space-y-0.5">
          <p className="text-xs font-semibold text-amber-300">
            This eval is NEVER shown to Claude.
          </p>
          <p className="text-xs text-amber-400/80">
            It is used by the verifier model only. Claude cannot see or game it.
            Write the criterion the verifier should check against Claude's summary.
          </p>
        </div>
      </div>

      {/* Eval textarea */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted-foreground font-medium">
          Completion criterion{' '}
          <span className="text-[9px] bg-orange-500/10 text-orange-400 border border-orange-500/20 px-1 py-0.5 rounded ml-1">
            verifier only
          </span>
        </label>
        <Textarea
          value={state.eval_text}
          onChange={(e) => onChange({ eval_text: e.target.value })}
          placeholder="e.g. All critical test failures are resolved or have a linked issue with a mitigation plan."
          className="min-h-32 text-sm resize-none"
        />
        <p className="text-[10px] text-muted-foreground">
          The verifier reads Claude's output and checks if this criterion is met.
        </p>
      </div>

      {/* Color picker */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted-foreground font-medium">Color</label>
        <div className="flex gap-2">
          {COLOR_SWATCHES.map((swatch) => (
            <button
              key={swatch.name}
              onClick={() => onChange({ color: swatch.name })}
              title={swatch.name}
              className={cn(
                'h-6 w-6 rounded-full transition-all',
                swatch.bg,
                state.color === swatch.name
                  ? `ring-2 ring-offset-2 ring-offset-background ${swatch.ring}`
                  : 'opacity-70 hover:opacity-100',
              )}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main wizard
// ---------------------------------------------------------------------------

export function EntryWizard({
  open,
  onOpenChange,
  projectId,
  initialType = 'cron',
  onCreated,
}: EntryWizardProps) {
  const [step, setStep] = useState(0);
  const [skills, setSkills] = useState<CronbanSkill[]>([]);
  const [columns, setColumns] = useState<KanbanColumn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [state, setState] = useState<WizardState>({
    entry_type: initialType,
    title: '',
    cron_expression: '',
    timezone: 'UTC',
    kanban_column_id: '',
    gate_check_interval_minutes: 30,
    target_cwd: '',
    create_new_session: false,
    skill_mode: 'inline',
    skill_id: '',
    prompt_text: '',
    eval_text: '',
    color: 'blue',
  });

  // Reset when dialog opens
  useEffect(() => {
    if (open) {
      setStep(0);
      setError(null);
      setState({
        entry_type: initialType,
        title: '',
        cron_expression: '',
        timezone: 'UTC',
        kanban_column_id: '',
        gate_check_interval_minutes: 30,
        target_cwd: '',
        create_new_session: false,
        skill_mode: 'inline',
        skill_id: '',
        prompt_text: '',
        eval_text: '',
        color: 'blue',
      });
    }
  }, [open, initialType]);

  // Load skills and columns on open
  useEffect(() => {
    if (!open) return;
    Promise.all([listSkills(projectId), listColumns(projectId)])
      .then(([s, c]) => {
        setSkills(s);
        setColumns(c);
      })
      .catch(() => {
        // Non-fatal: wizard still works, dropdowns will just be empty
      });
  }, [open, projectId]);

  const patch = useCallback((p: Partial<WizardState>) => {
    setState((prev) => ({ ...prev, ...p }));
  }, []);

  // Validation per step
  const canAdvance = useCallback((): boolean => {
    if (step === 0) {
      if (!state.title.trim()) return false;
      if (state.entry_type === 'cron' && !state.cron_expression.trim()) return false;
      return true;
    }
    if (step === 1) {
      if (state.skill_mode === 'saved' && !state.skill_id) return false;
      if (state.skill_mode === 'inline' && !state.prompt_text.trim()) return false;
      return true;
    }
    return true;
  }, [step, state]);

  const handleNext = useCallback(() => {
    if (step < STEPS.length - 1) setStep((s) => s + 1);
  }, [step]);

  const handleBack = useCallback(() => {
    if (step > 0) setStep((s) => s - 1);
  }, [step]);

  const handleCreate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload: Parameters<typeof createEntry>[0] = {
        project_id: projectId ?? null,
        title: state.title.trim(),
        entry_type: state.entry_type,
        color: state.color,
        target_cwd: state.target_cwd.trim() || null,
        create_new_session: state.create_new_session,
        timezone: state.timezone,
        eval_text: state.eval_text.trim() || undefined,
      };

      if (state.skill_mode === 'saved' && state.skill_id) {
        payload.skill_id = state.skill_id;
      } else if (state.prompt_text.trim()) {
        payload.prompt_text = state.prompt_text.trim();
      }

      if (state.entry_type === 'cron') {
        payload.cron_expression = state.cron_expression.trim();
      } else if (state.entry_type === 'gate') {
        payload.kanban_column_id = state.kanban_column_id || null;
        payload.gate_check_interval_minutes = state.gate_check_interval_minutes;
      }

      const entry = await createEntry(payload);
      onCreated(entry);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create entry');
    } finally {
      setLoading(false);
    }
  }, [state, projectId, onCreated, onOpenChange]);

  const isLastStep = step === STEPS.length - 1;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="text-base">
              {step === 0 && 'New Entry — Schedule'}
              {step === 1 && 'New Entry — Skill'}
              {step === 2 && 'New Entry — Eval'}
            </DialogTitle>
            <StepIndicator current={step} total={STEPS.length} />
          </div>
        </DialogHeader>

        {/* Step content */}
        <div className="py-2 overflow-y-auto max-h-[60vh]">
          {step === 0 && (
            <Step1Schedule state={state} onChange={patch} columns={columns} />
          )}
          {step === 1 && (
            <Step2Skill state={state} onChange={patch} skills={skills} />
          )}
          {step === 2 && (
            <Step3Eval state={state} onChange={patch} />
          )}
        </div>

        {/* Error */}
        {error && (
          <p className="text-xs text-destructive border border-destructive/30 rounded px-2 py-1.5 bg-destructive/5">
            {error}
          </p>
        )}

        <DialogFooter className="gap-2 sm:gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleBack}
            disabled={step === 0 || loading}
            className="gap-1"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
            Back
          </Button>

          {!isLastStep ? (
            <Button
              size="sm"
              onClick={handleNext}
              disabled={!canAdvance()}
              className="gap-1"
            >
              Next
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          ) : (
            <Button
              size="sm"
              onClick={handleCreate}
              disabled={loading}
              className="gap-1"
            >
              {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              {loading ? 'Creating…' : 'Create Entry'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
