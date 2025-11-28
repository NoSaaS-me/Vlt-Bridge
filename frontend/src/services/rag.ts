import { ChatRequest, ChatResponse, StatusResponse } from '@/types/rag';
import { apiFetch } from './api';

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  return apiFetch<ChatResponse>('/api/rag/chat', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getRagStatus(): Promise<StatusResponse> {
  return apiFetch<StatusResponse>('/api/rag/status');
}