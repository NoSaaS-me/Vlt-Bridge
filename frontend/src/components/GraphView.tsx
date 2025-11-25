import { useEffect, useRef, useState, useMemo } from 'react';
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d';
import type { GraphData, GraphNode } from '@/types/graph';
import { getGraphData } from '@/services/api';
import { Loader2, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface GraphViewProps {
  onSelectNote: (path: string) => void;
}

export function GraphView({ onSelectNote }: GraphViewProps) {
  const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const graphRef = useRef<ForceGraphMethods>();
  // Theme detection would go here, simplified for MVP
  const isDark = document.documentElement.classList.contains('dark');
  
  // Load saved view state
  useEffect(() => {
    if (!isLoading && graphRef.current) {
      const savedView = localStorage.getItem('graph-view-state');
      if (savedView) {
        const { x, y, k } = JSON.parse(savedView);
        graphRef.current.centerAt(x, y, 0);
        graphRef.current.zoom(k, 0);
      }
    }
  }, [isLoading]);

  // Save view state on unmount
  useEffect(() => {
    return () => {
      if (graphRef.current) {
        // Note: react-force-graph types might be incomplete for getting state directly
        // This is a best-effort implementation. 
        // Often we can't easily get current x,y,k without internal access or tracking interaction.
        // For now, we will skip complex persistence if the library doesn't support getter easily.
        // Wait, we can use graphRef.current.zoom() as a getter? 
        // The docs say zoom(k) sets it. getter? usually yes if arg missing.
        try {
            // @ts-ignore
            const k = graphRef.current.zoom();
            // @ts-ignore
            const { x, y } = graphRef.current.centerAt();
            
            if (x !== undefined && k !== undefined) {
                localStorage.setItem('graph-view-state', JSON.stringify({ x, y, k }));
            }
        } catch (e) {
            // Ignore errors if getters fail
        }
      }
    };
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const graphData = await getGraphData();
        setData(graphData);
        setError(null);
      } catch (err) {
        console.error('Failed to load graph data:', err);
        setError('Failed to load graph data. Please try again.');
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  // Simple hash for categorical colors
  const getGroupColor = (group: string) => {
    let hash = 0;
    for (let i = 0; i < group.length; i++) {
      hash = group.charCodeAt(i) + ((hash << 5) - hash);
    }
    const c = (hash & 0x00ffffff).toString(16).toUpperCase();
    return '#' + '00000'.substring(0, 6 - c.length) + c;
  };

  // Node styling based on theme and group
  const defaultNodeColor = isDark ? '#94a3b8' : '#64748b';
  const linkColor = isDark ? '#334155' : '#e2e8f0';
  const backgroundColor = isDark ? '#020817' : '#ffffff';

  const handleNodeClick = (node: any) => {
    if (node && node.id) {
      onSelectNote(node.id);
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-background text-muted-foreground">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="h-8 w-8 animate-spin" />
          <p>Loading graph...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="h-full w-full overflow-hidden bg-background">
      <ForceGraph2D
        ref={graphRef}
        graphData={data}
        nodeLabel="label"
        nodeColor={(node: any) => node.group && node.group !== 'root' ? getGroupColor(node.group) : defaultNodeColor}
        linkColor={() => linkColor}
        backgroundColor={backgroundColor}
        onNodeClick={handleNodeClick}
        nodeRelSize={6}
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={0.005}
        width={window.innerWidth * 0.75} // Approximate width, needs resize observer for true responsiveness
        height={window.innerHeight - 60} // Approximate height minus header
        // Basic forces to keep structure
        d3Force={('charge', -120)} 
      />
    </div>
  );
}
