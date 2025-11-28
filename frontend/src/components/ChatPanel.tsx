import { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ChatMessage } from './ChatMessage';
import { sendChat } from '@/services/rag';
import { ChatMessage as ChatMessageType } from '@/types/rag';
import { useToast } from '@/hooks/useToast';

interface ChatPanelProps {
  onNavigateToNote: (path: string) => void;
}

export function ChatPanel({ onNavigateToNote }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const toast = useToast();

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: ChatMessageType = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      // Create request with full history (US3 prep)
      const history = [...messages, userMsg];
      const response = await sendChat({ messages: history });
      
      const assistantMsg: ChatMessageType = {
        role: 'assistant',
        content: response.answer,
        timestamp: new Date().toISOString(),
        sources: response.sources,
        notes_written: response.notes_written
      };
      
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      console.error("Chat error:", err);
      toast.error("Failed to get response from agent");
      // Optionally remove user message or show error state
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <h2 className="font-semibold">Gemini Planning Agent</h2>
        <p className="text-xs text-muted-foreground">Ask questions about your vault</p>
      </div>

      {/* Message List */}
      <div className="flex-1 overflow-y-auto" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-8 text-center">
            <p>👋 Hi! I can help you navigate this vault.</p>
            <p className="text-sm mt-2">Try asking: "How does authentication work?"</p>
          </div>
        ) : (
          <div className="divide-y divide-border/50">
            {messages.map((msg, i) => (
              <ChatMessage key={i} message={msg} onSourceClick={onNavigateToNote} />
            ))}
            {isLoading && (
              <div className="p-4 flex items-center gap-2 text-muted-foreground text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                Thinking...
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-border">
        <div className="flex gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            className="min-h-[40px] max-h-[150px] resize-none"
            rows={1}
          />
          <Button onClick={handleSubmit} disabled={isLoading || !input.trim()} size="icon">
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
