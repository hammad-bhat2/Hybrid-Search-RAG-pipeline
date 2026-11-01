import React, { useState, useEffect } from 'react';
import { MessageSquare, UploadCloud, Database, Cpu, Activity, ShieldCheck } from 'lucide-react';
import ChatView from './components/ChatView.jsx';
import UploadView from './components/UploadView.jsx';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [health, setHealth] = useState(null);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/v1/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (err) {
      console.warn('Backend not yet reachable:', err.message);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      {/* App Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">⚡</div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="brand-title">Hybrid Search RAG</span>
              <span className="brand-badge">Production Ready</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <MessageSquare size={16} />
            <span>Chat & Retrieval</span>
          </button>
          <button
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <UploadCloud size={16} />
            <span>Documents & Ingestion</span>
          </button>
        </nav>

        {/* System Health Indicators */}
        <div className="header-status">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span
              className="status-indicator"
              style={{
                backgroundColor: health?.status === 'healthy' ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                boxShadow: `0 0 8px ${health?.status === 'healthy' ? 'var(--accent-emerald)' : 'var(--accent-rose)'}`
              }}
            />
            <span style={{ fontSize: '0.75rem', fontWeight: 500 }}>
              {health?.status === 'healthy' ? 'Online' : 'Connecting...'}
            </span>
          </div>

          {health && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: 'rgba(255, 255, 255, 0.05)',
              padding: '2px 8px',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.7rem'
            }}>
              <Database size={11} color="#60a5fa" />
              <span>{health.qdrant?.points_count ?? 0} vectors</span>
            </div>
          )}

          {health?.llm_provider && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: 'rgba(255, 255, 255, 0.05)',
              padding: '2px 8px',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.7rem',
              color: '#a78bfa'
            }}>
              <Cpu size={11} />
              <span>{health.llm_provider}</span>
            </div>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'chat' ? (
          <ChatView />
        ) : (
          <UploadView onDocumentUploaded={checkHealth} />
        )}
      </main>
    </div>
  );
}
