/**
 * useSessionPolling — Polls daemon for sessions + hook events.
 *
 * Extracts the polling state/effects from the old AgentsPage monolith.
 * Sessions poll every 5s, events every 3s.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import {
  listSessions,
  listHookEvents,
  type AgentSession,
  type HookEvent,
} from '@/services/daemon-api';

export interface SessionPollingState {
  sessions: AgentSession[];
  events: HookEvent[];
  isLoading: boolean;
  daemonOnline: boolean;
  lastRefresh: Date;
  refresh: () => void;
}

export function useSessionPolling(): SessionPollingState {
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [events, setEvents] = useState<HookEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [daemonOnline, setDaemonOnline] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const pollRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const eventPollRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const fetchSessions = useCallback(async () => {
    try {
      const data = await listSessions();
      setSessions(data);
      setDaemonOnline(true);
      setLastRefresh(new Date());
    } catch {
      setDaemonOnline(false);
      setSessions([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchEvents = useCallback(async () => {
    try {
      const data = await listHookEvents(200);
      setEvents(data);
    } catch {
      // daemon offline — silently ignore
    }
  }, []);

  const refresh = useCallback(() => {
    fetchSessions();
    fetchEvents();
  }, [fetchSessions, fetchEvents]);

  useEffect(() => {
    fetchSessions();
    fetchEvents();
    pollRef.current = setInterval(fetchSessions, 5000);
    eventPollRef.current = setInterval(fetchEvents, 3000);
    return () => {
      clearInterval(pollRef.current);
      clearInterval(eventPollRef.current);
    };
  }, [fetchSessions, fetchEvents]);

  return { sessions, events, isLoading, daemonOnline, lastRefresh, refresh };
}
