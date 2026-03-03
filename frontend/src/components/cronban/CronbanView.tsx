/**
 * CronbanView — Pipeline-based scheduling and workflow automation UI.
 *
 * Five tabs:
 *   Calendar  — time-based CronTriggers (cron/rrule)
 *   Kanban    — pipeline workflows (stages + cards)
 *   Webhooks  — external event listeners
 *   Skills    — reusable prompt template library
 *   Gates     — reusable eval criteria (helper Claude)
 */
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Calendar, Kanban, Webhook, BookOpen, ShieldCheck } from 'lucide-react';
import { SkillsPanel } from './SkillsPanel';
import { GatesPanel } from './GatesPanel';
import { CalendarView } from './CalendarView';
import { KanbanBoard } from './KanbanBoard';
import { WebhooksPanel } from './WebhooksPanel';

export function CronbanView({ projectId }: { projectId?: string }) {
  return (
    <Tabs defaultValue="kanban" className="h-full flex flex-col">
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
          value="webhooks"
          className="h-8 gap-1.5 text-xs data-[state=active]:bg-muted"
        >
          <Webhook className="h-3.5 w-3.5" />
          Webhooks
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
        <KanbanBoard projectId={projectId} />
      </TabsContent>

      <TabsContent value="webhooks" className="flex-1 overflow-hidden mt-0">
        <WebhooksPanel projectId={projectId} />
      </TabsContent>

      <TabsContent value="skills" className="flex-1 overflow-hidden mt-0">
        <SkillsPanel projectId={projectId} />
      </TabsContent>

      <TabsContent value="gates" className="flex-1 overflow-hidden mt-0">
        <GatesPanel projectId={projectId} />
      </TabsContent>
    </Tabs>
  );
}
