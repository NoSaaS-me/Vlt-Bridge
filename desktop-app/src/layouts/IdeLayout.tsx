import React, { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Terminal } from '../components/Terminal';
import { SessionStatus } from '../components/SessionStatus';
import { Folder, FileCode } from 'lucide-react';

export const IdeLayout: React.FC = () => {
  const [files, setFiles] = useState<string[]>([]);

  useEffect(() => {
    // Call native Rust command to list current directory
    invoke<string[]>('list_directory', { path: './' })
      .then(setFiles)
      .catch(console.error);
  }, []);

  return (
    <div className="flex h-screen bg-gray-900 text-white overflow-hidden">
      {/* Sidebar: Native File System Access */}
      <div className="w-64 border-r border-gray-800 p-4 flex flex-col">
        <h2 className="text-sm font-bold uppercase tracking-wider text-gray-500 mb-4 flex items-center gap-2">
          <Folder size={14} /> Explorer
        </h2>
        <div className="flex-1 overflow-y-auto space-y-1">
          {files.map((file) => (
            <div key={file} className="flex items-center gap-2 text-sm text-gray-300 hover:bg-gray-800 px-2 py-1 rounded cursor-pointer transition-colors">
              <FileCode size={14} className="text-blue-400" />
              {file}
            </div>
          ))}
        </div>
      </div>
      
      <div className="flex-1 flex flex-col">
        <header className="h-12 border-b border-gray-800 flex items-center justify-between px-4 bg-gray-900/50 backdrop-blur">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Vlt Desktop</span>
            <span className="text-xs text-gray-700">/</span>
            <span className="text-sm font-mono text-blue-400">main.py</span>
          </div>
          <SessionStatus status="READY" />
        </header>
        
        <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
          <div className="flex-1 border-r border-gray-800 relative bg-black/20">
            {/* Editor Placeholder */}
            <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-700">
              <p className="text-xl font-light mb-2">IDE 2.0</p>
              <p className="text-xs font-mono uppercase tracking-widest opacity-50">Editor View Active</p>
            </div>
          </div>
          
          <div className="w-full md:w-1/3 h-64 md:h-full bg-black">
            <Terminal onInput={(d) => console.log(d)} />
          </div>
        </div>
      </div>
    </div>
  );
};