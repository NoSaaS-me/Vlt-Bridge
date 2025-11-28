import React, { useEffect, useState, Component, type ErrorInfo } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import { NoteViewer } from '@/components/NoteViewer';
import { SearchWidget } from '@/components/SearchWidget';
import type { Note } from '@/types/note';
import type { SearchResult } from '@/types/search';
import { getNote } from '@/services/api';
import { Loader2, AlertTriangle } from 'lucide-react';

// Mock window.openai for development
if (!window.openai) {
  console.warn("Mocking window.openai (dev mode)");
  window.openai = {
    toolOutput: null
  };
}

// Extend window interface
declare global {
  interface Window {
    openai: {
      toolOutput: any;
      toolInput?: any;
    };
  }
}

class WidgetErrorBoundary extends Component<{ children: React.ReactNode }, { hasError: boolean, error: Error | null }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Widget Error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background text-destructive p-4 flex flex-col items-center justify-center text-center">
          <AlertTriangle className="h-8 w-8 mb-2" />
          <h2 className="font-bold text-lg">Something went wrong</h2>
          <p className="text-sm text-muted-foreground mt-1 max-w-xs">
            {this.state.error?.message || "The widget could not be displayed."}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

const WidgetApp = () => {
  const [view, setView] = useState<'loading' | 'note' | 'search' | 'error'>('loading');
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Function to check for data
    const checkData = () => {
      const toolOutput = window.openai.toolOutput;
      if (toolOutput) {
        console.log("Tool output found:", toolOutput);
        if (toolOutput.note) {
          setView('note');
          setData(toolOutput.note);
        } else if (toolOutput.results) {
          setView('search');
          setData(toolOutput.results);
        } else {
          console.warn("Unknown tool output format", toolOutput);
          setError("Unknown content format.");
          setView('error');
        }
        return true;
      }
      return false;
    };

    // Check immediately
    if (checkData()) return;

    // If not found, check if we have toolInput (loading state)
    // Note: window.openai types might vary, we check safely
    const toolInput = (window.openai as any).toolInput;
    if (toolInput) {
      console.log("Tool input present, waiting for output:", toolInput);
      setView('loading');
    } else {
       // Fallback for dev testing via URL (only if no toolInput either)
        const params = new URLSearchParams(window.location.search);
        const mockType = params.get('type');
        if (mockType === 'note') {
            // ... existing mock logic ...
            setView('note');
            setData({
                title: "Demo Note",
                note_path: "demo.md",
                body: "# Demo Note\n\nMock note.",
                version: 1,
                size_bytes: 100,
                created: new Date().toISOString(),
                updated: new Date().toISOString(),
                metadata: {}
            });
            return;
        } else if (mockType === 'search') {
            setView('search');
            setData([{ title: "Result 1", note_path: "res1.md", snippet: "Match", score: 1, updated: new Date() }]);
            return;
        }
        
        // If absolutely nothing, show error (or keep loading if we suspect latency)
        // But sticking to 'loading' is safer than error if we are polling.
    }

    // Poll for data injection
    const interval = setInterval(() => {
       if (checkData()) clearInterval(interval);
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const handleWikilinkClick = (linkText: string) => {
    console.log("Clicked wikilink in widget:", linkText);
    // Convert wikilink text to path (simple slugification or just try reading it)
    // We can try to fetch it directly.
    // Reuse handleNoteSelect logic but we need to slugify first?
    // For now, let's try strict filename matching or just pass text.
    // getNote expects a path.
    // We'll just alert for now or try to resolve it.
    // Let's try to fetch it as a path (assuming user clicked "Getting Started")
    handleNoteSelect(linkText + ".md"); 
  };

  const handleNoteSelect = async (path: string) => {
     console.log("Selected note:", path);
     setView('loading');
     try {
       const note = await getNote(path);
       setData(note);
       setView('note');
     } catch (err: any) {
       console.error("Failed to load note:", err);
       const msg = err?.message || String(err);
       const status = err?.status ? ` (Status: ${err.status})` : '';
       setError(`Failed to load note: ${path}\nError: ${msg}${status}`);
       setView('error');
     }
  };

  return (
    <div className="dark min-h-screen bg-background text-foreground flex flex-col">
        {view === 'loading' && (
          <div className="flex-1 flex flex-col items-center justify-center p-4 text-center">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground mb-4" />
            <p className="text-sm text-muted-foreground">
              {window.openai?.toolInput?.query 
                ? `Searching for "${window.openai.toolInput.query}"...`
                : "Waiting for tool execution..."}
            </p>
            <p className="text-xs text-muted-foreground mt-2">
              (Please confirm the action in ChatGPT)
            </p>
          </div>
        )}

        {view === 'error' && (
          <div className="p-4 text-destructive flex flex-col gap-4 overflow-auto">
            <div>
              <h2 className="font-bold">Error</h2>
              <p>{error}</p>
            </div>
            
            <div className="bg-muted p-2 rounded text-xs font-mono text-foreground">
              <p className="font-bold mb-1">Debug Info (window.openai):</p>
              <pre className="whitespace-pre-wrap break-all">
                {JSON.stringify(window.openai, null, 2)}
              </pre>
            </div>

            <button 
              className="px-4 py-2 bg-primary text-primary-foreground rounded hover:opacity-90"
              onClick={() => window.location.reload()}
            >
              Reload Widget
            </button>
          </div>
        )}

        {view === 'note' && data && (
          <NoteViewer 
            note={data as Note} 
            backlinks={[]} // Backlinks usually fetched separately, omit for V1 widget
            onWikilinkClick={handleWikilinkClick} 
          />
        )}

        {view === 'search' && data && (
          <SearchWidget 
            results={data as SearchResult[]} 
            onSelectNote={handleNoteSelect} 
          />
        )}
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <WidgetErrorBoundary>
      <WidgetApp />
    </WidgetErrorBoundary>
  </React.StrictMode>
);
