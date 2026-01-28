import React, { useState } from 'react';

export const Settings: React.FC = () => {
  const [url, setUrl] = useState('ws://localhost:8000');
  const [token, setToken] = useState('');

  const handleSave = () => {
    localStorage.setItem('daemon_url', url);
    localStorage.setItem('daemon_token', token);
    alert('Settings saved');
  };

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Daemon URL</label>
          <input 
            type="text" 
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="w-full p-2 border rounded bg-gray-50"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Auth Token</label>
          <input 
            type="password" 
            value={token}
            onChange={(e) => setToken(e.target.value)}
            className="w-full p-2 border rounded bg-gray-50"
          />
        </div>

        <button 
          onClick={handleSave}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Save Connection
        </button>
      </div>
    </div>
  );
};
