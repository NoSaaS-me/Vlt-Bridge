import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { getLogs, type LogEntry } from '@/services/api';
import { RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function SystemLogs() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchLogs = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getLogs();
      // Sort logs newest first if backend returns oldest first
      setLogs(data.reverse()); 
    } catch (err) {
      console.error('Failed to fetch logs:', err);
      setError('Failed to load logs.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    // Optional: Auto-refresh every 5 seconds?
    // const interval = setInterval(fetchLogs, 5000);
    // return () => clearInterval(interval);
  }, []);

  const getLevelVariant = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR': return 'destructive';
      case 'WARNING': return 'secondary'; // yellow-ish usually
      case 'INFO': return 'default';
      case 'DEBUG': return 'outline';
      default: return 'secondary';
    }
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>System Logs</CardTitle>
            <CardDescription>Backend operational logs (last 100 entries)</CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={fetchLogs} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-[400px] w-full p-4 pt-0">
          {error && <div className="text-destructive text-sm p-2">{error}</div>}
          {logs.length === 0 && !isLoading && !error && (
            <div className="text-muted-foreground text-sm p-2">No logs available.</div>
          )}
          <div className="space-y-2">
            {logs.map((log, i) => (
              <div key={i} className="text-xs border-b border-border pb-2 last:border-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-muted-foreground font-mono">{new Date(log.timestamp).toLocaleTimeString()}</span>
                  <Badge variant={getLevelVariant(log.level)} className="h-5 px-1.5 text-[10px]">
                    {log.level}
                  </Badge>
                </div>
                <div className="font-mono break-all whitespace-pre-wrap text-foreground/90">
                  {log.message}
                </div>
                {log.extra && Object.keys(log.extra).length > 0 && (
                  <div className="mt-1 pl-2 border-l-2 border-muted">
                    <pre className="text-[10px] text-muted-foreground">
                      {JSON.stringify(log.extra, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
