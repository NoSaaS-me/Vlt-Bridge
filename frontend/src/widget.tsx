import React, { useEffect, useState, Component, type ErrorInfo } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import { NoteViewer } from '@/components/NoteViewer';
import { SearchWidget } from '@/components/SearchWidget';
import type { Note } from '@/types/note';
import type { SearchResult } => '@/types/search';
import { Loader2, AlertTriangle } from 'lucide-react';

// Mock window.openai for development
if (!window.openai) {
  window.openai = {
    toolOutput: null
  };
}

// Extend window interface
declare global {
  interface Window {
    openai: {
      toolOutput: any;
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
    // In a real ChatGPT app, toolOutput is injected before script execution or available on load.
    // We check it here.
    const toolOutput = window.openai.toolOutput;
    console.log("Widget loaded with toolOutput:", toolOutput);

    if (!toolOutput) {
      // Fallback for dev testing via URL
      const params = new URLSearchParams(window.location.search);
      const mockType = params.get('type');
      if (mockType === 'note') {
        // Mock note data
        setView('note');
        setData({
          title: "Demo Note",
          note_path: "demo.md",
          body: "# Demo Note\n\nThis is a **markdown** note rendered in the widget.",
          version: 1,
          size_bytes: 100,
          created: new Date().toISOString(),
          updated: new Date().toISOString(),
          metadata: {}
        });
      } else if (mockType === 'search') {
        setView('search');
        setData([
          { title: "Result 1", note_path: "res1.md", snippet: "Found match...", score: 1.0, updated: new Date() }
        ]);
      } else {
        setError("No content data found. (window.openai.toolOutput is empty)");
        setView('error');
      }
      return;
    }

    // Detect content type based on shape
    if (toolOutput.note) {
      setView('note');
      setData(toolOutput.note);
    } else if (toolOutput.results) {
      setView('search');
      setData(toolOutput.results);
    } else {
      // Fallback: try to guess or show raw
      console.warn("Unknown tool output format", toolOutput);
      setError("Unknown content format.");
      setView('error');
    }
  }, []);

  const handleWikilinkClick = (linkText: string) => {
    console.log("Clicked wikilink in widget:", linkText);
    alert(`Navigation to "${linkText}" requested. (Widget navigation pending implementation)`);
  };

  const handleNoteSelect = (path: string) => {
     console.log("Selected note from search:", path);
     alert(`Opening "${path}"... (Widget navigation pending implementation)`);
  };

  return (
    <div className="dark min-h-screen bg-background text-foreground flex flex-col">
        {view === 'loading' && (
          <div className="flex-1 flex items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )}

        {view === 'error' && (
          <div className="p-4 text-destructive">
            <h2 className="font-bold">Error</h2>
            <p>{error}</p>
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
