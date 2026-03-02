/**
 * CronbanView — Cron + Kanban + Calendar scheduling UI.
 *
 * Three tabs:
 *   Calendar  — time-based scheduled jobs (cron/rrule)
 *   Kanban    — gate-based cards (AI verifier enforces eval criterion)
 *   Skills    — reusable prompt template library
 *
 * Three-part job structure (enforced everywhere in UI):
 *   1. Schedule  — when to fire
 *   2. Skill     — what Claude sees
 *   3. Eval      — hidden criterion Claude never sees
 */
import { useState, useCallback } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Calendar, Kanban, BookOpen, ShieldCheck } from 'lucide-react';
import { SkillsPanel } from './SkillsPanel';
import { GatesPanel } from './GatesPanel';
import { CalendarView } from './CalendarView';
import { KanbanBoard } from './KanbanBoard';
import { EntryWizard } from './EntryWizard';

export function CronbanView({ projectId }: { projectId?: string }) {
  const [wizardOpen, setWizardOpen] = useState(false);
  const [wizardColumnId, setWizardColumnId] = useState<string | undefined>(undefined);
  const [boardRefreshKey, setBoardRefreshKey] = useState(0);

  const handleNewEntry = useCallback((columnId: string) => {
    setWizardColumnId(columnId);
    setWizardOpen(true);
  }, []);

  return (
    <>
      <Tabs defaultValue="skills" className="h-full flex flex-col">
        <TabsList className="shrink-0 rounded-none border-b border-border bg-transparent px-4 justify-start gap-1 h-10">
          <TabsTrigger
            value="calendar"
            className="h-8 gap-1.5 text-xs data-[state=active]:bg-muted"
          >
            <Calendar className="h-3.5 w-3.5" />
            Calendar
          </TabsTrigger>
          <TabsTrigger
            value="kanban"
            className="h-8 gap-1.5 text-xs data-[state=active]:bg-muted"
          >
            <Kanban className="h-3.5 w-3.5" />
            Kanban
          </TabsTrigger>
          <TabsTrigger
            value="skills"
            className="h-8 gap-1.5 text-xs data-[state=active]:bg-muted"
          >
            <BookOpen className="h-3.5 w-3.5" />
            Skills
          </TabsTrigger>
          <TabsTrigger
            value="gates"
            className="h-8 gap-1.5 text-xs data-[state=active]:bg-muted"
          >
            <ShieldCheck className="h-3.5 w-3.5" />
            Gates
          </TabsTrigger>
        </TabsList>

        <TabsContent value="calendar" className="flex-1 overflow-hidden mt-0">
          <CalendarView projectId={projectId} />
        </TabsContent>

        <TabsContent value="kanban" className="flex-1 overflow-hidden mt-0">
          <KanbanBoard projectId={projectId} onNewEntry={handleNewEntry} refreshKey={boardRefreshKey} />
        </TabsContent>

        <TabsContent value="skills" className="flex-1 overflow-hidden mt-0">
          <SkillsPanel projectId={projectId} />
        </TabsContent>

        <TabsContent value="gates" className="flex-1 overflow-hidden mt-0">
          <GatesPanel projectId={projectId} />
        </TabsContent>
      </Tabs>

      {/* Shared entry creation wizard — gate type pre-selected when opened from Kanban */}
      <EntryWizard
        open={wizardOpen}
        onOpenChange={(open) => {
          setWizardOpen(open);
          if (!open) setWizardColumnId(undefined);
        }}
        projectId={projectId}
        initialType={wizardColumnId ? 'gate' : 'cron'}
        onCreated={() => {
          setWizardOpen(false);
          setWizardColumnId(undefined);
          setBoardRefreshKey((k) => k + 1);
        }}
      />
    </>
  );
}
