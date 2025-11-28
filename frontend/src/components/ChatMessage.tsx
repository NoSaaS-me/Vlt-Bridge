import { ChatMessage as ChatMessageType } from '@/types/rag';
import { cn } from '@/lib/utils';
import { User, Bot } from 'lucide-react';
import { SourceList } from './SourceList';

interface ChatMessageProps {
  message: ChatMessageType;
  onSourceClick: (path: string) => void;
}

export function ChatMessage({ message, onSourceClick }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn("flex gap-3 p-4", isUser ? "bg-transparent" : "bg-muted/30")}>
      <div className={cn(
        "h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0",
        isUser ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
      )}>
        {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
      </div>
      
      <div className="flex-1 space-y-2 overflow-hidden">
        <div className="prose dark:prose-invert text-sm max-w-none whitespace-pre-wrap">
          {message.content}
        </div>
        
        {!isUser && message.sources && (
          <SourceList sources={message.sources} onSourceClick={onSourceClick} />
        )}
      </div>
    </div>
  );
}
