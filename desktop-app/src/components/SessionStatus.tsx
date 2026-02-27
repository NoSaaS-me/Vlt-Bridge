import React from 'react';
import { Circle, Play, Pause, AlertCircle } from 'lucide-react';

interface SessionStatusProps {
  status: 'STARTING' | 'READY' | 'RUNNING' | 'PAUSED' | 'ERROR';
}

export const SessionStatus: React.FC<SessionStatusProps> = ({ status }) => {
  const getIcon = () => {
    switch (status) {
      case 'STARTING': return <Circle className="animate-pulse text-yellow-500" size={16} />;
      case 'READY': return <Circle className="text-green-500" size={16} />;
      case 'RUNNING': return <Play className="text-green-500" size={16} />;
      case 'PAUSED': return <Pause className="text-gray-500" size={16} />;
      case 'ERROR': return <AlertCircle className="text-red-500" size={16} />;
    }
  };

  return (
    <div className="flex items-center gap-2 px-3 py-1 rounded bg-gray-800 border border-gray-700">
      {getIcon()}
      <span className="text-xs font-mono text-gray-300">{status}</span>
    </div>
  );
};
