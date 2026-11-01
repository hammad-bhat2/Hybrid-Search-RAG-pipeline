import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, BookOpen, Layers, ChevronDown, ChevronUp, Cpu, SlidersHorizontal, Info } from 'lucide-react';

export default function ChatView() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Welcome! I am your Hybrid Search RAG assistant. Ask me questions about your indexed documents, and I will retrieve grounded answers using dense vector semantics and sparse BM25 keyword matching.',
      sources: []
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [expandedSources, setExpandedSources] = useState({});
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [latestChunks, setLatestChunks] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e?.preventDefault();
    const queryText = input.trim();
    if (!queryText || loading) return;

    setInput('');
    const newMessages = [
      ...messages,
      { role: 'user', content: queryText }
    ];
    setMessages(newMessages);
    setLoading(true);

    try {
      const res = await fetch('/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, top_k: topK })
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      setLatestChunks(data.sources || []);

      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          content: data.answer,
          sources: data.sources || [],
          provider: data.provider
        }
      ]);
    } catch (err) {
      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          content: `⚠️ Failed to generate answer: ${err.message}. Please ensure documents are uploaded and backend is running.`,
          sources: []
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const toggleSourceExpand = (msgIndex, srcIndex) => {
    const key = `${msgIndex}_${srcIndex}`;
    setExpandedSources(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const starterQueries = [
    "What are the main concepts in the uploaded document?",
    "Summarize the technical architecture and specifications.",
    "List the key requirements and constraints mentioned."
  ];

  return (
    <div className="chat-layout">
      {/* Main Chat Panel */}
      <div className="chat-main">
        {/* Chat Control Header */}
        <div style={{
          padding: '0.85rem 1.25rem',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'rgba(10, 15, 29, 0.6)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
            <Cpu size={16} color="#3b82f6" />
            <span style={{ fontWeight: 600 }}>RAG Chat Conversation</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <SlidersHorizontal size={14} />
              <span>Top-K:</span>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{
                  background: 'rgba(31, 41, 55, 0.8)',
                  color: 'white',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '2px 6px',
                  fontSize: '0.8rem',
                  outline: 'none'
                }}
              >
                <option value={3}>3 Chunks</option>
                <option value={5}>5 Chunks</option>
                <option value={8}>8 Chunks</option>
              </select>
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
              onClick={() => setInspectorOpen(!inspectorOpen)}
            >
              {inspectorOpen ? 'Hide Inspector' : 'Show RRF Inspector'}
            </button>
          </div>
        </div>

        {/* Message Stream */}
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`message-bubble ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', opacity: 0.8 }}>
                {msg.role === 'user' ? <User size={14} /> : <Bot size={14} color="#60a5fa" />}
                <span style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  {msg.role === 'user' ? 'You' : 'Hybrid RAG Assistant'}
                </span>
                {msg.provider && (
                  <span style={{
                    marginLeft: 'auto',
                    fontSize: '0.65rem',
                    padding: '1px 6px',
                    borderRadius: '4px',
                    background: 'rgba(255, 255, 255, 0.1)',
                    color: 'var(--text-muted)'
                  }}>
                    {msg.provider}
                  </span>
                )}
              </div>

              {/* Message text with basic whitespace preservation */}
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                {msg.content}
              </div>

              {/* Source Citations */}
              {msg.sources && msg.sources.length > 0 && (
                <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <BookOpen size={13} />
                    <span>Referenced Citations ({msg.sources.length})</span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {msg.sources.map((src, sIdx) => {
                      const isExpanded = expandedSources[`${idx}_${sIdx}`];
                      return (
                        <div
                          key={sIdx}
                          style={{
                            background: 'rgba(0, 0, 0, 0.2)',
                            border: '1px solid var(--border-color)',
                            borderRadius: 'var(--radius-sm)',
                            padding: '0.5rem 0.75rem',
                            fontSize: '0.8rem'
                          }}
                        >
                          <div
                            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                            onClick={() => toggleSourceExpand(idx, sIdx)}
                          >
                            <span style={{ fontWeight: 600, color: '#93c5fd' }}>
                              [{sIdx + 1}] {src.file_name} (Page {src.page})
                            </span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              {src.rrf_score && (
                                <span className="code-font" style={{ fontSize: '0.7rem', color: '#34d399' }}>
                                  RRF: {src.rrf_score.toFixed(4)}
                                </span>
                              )}
                              {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            </div>
                          </div>

                          {isExpanded && (
                            <div style={{
                              marginTop: '0.5rem',
                              paddingTop: '0.5rem',
                              borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                              fontSize: '0.75rem',
                              color: 'var(--text-muted)',
                              lineHeight: '1.5'
                            }}>
                              {src.text}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="message-bubble message-assistant">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--text-muted)' }}>
                <Sparkles size={16} color="#60a5fa" className="spin" />
                <span>Performing hybrid dense & sparse vector retrieval and synthesizing grounded response...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Suggested Queries if few messages */}
        {messages.length <= 2 && (
          <div style={{ padding: '0.75rem 1.25rem', display: 'flex', gap: '0.5rem', overflowX: 'auto', background: 'rgba(10, 15, 29, 0.4)' }}>
            {starterQueries.map((query, qIdx) => (
              <button
                key={qIdx}
                className="btn btn-secondary"
                style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', whiteSpace: 'nowrap' }}
                onClick={() => { setInput(query); }}
              >
                {query}
              </button>
            ))}
          </div>
        )}

        {/* Input Bar */}
        <form onSubmit={handleSend} className="chat-input-bar">
          <input
            type="text"
            className="chat-input"
            placeholder="Ask a question about your indexed documents..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="btn btn-primary" disabled={loading || !input.trim()}>
            <Send size={15} />
            <span>Send</span>
          </button>
        </form>
      </div>

      {/* RRF Transparency Inspector Side Panel */}
      {inspectorOpen && (
        <div className="inspector-panel">
          <div className="inspector-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={16} color="#8b5cf6" />
              <span>Hybrid Search Inspector</span>
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>RRF (k=60)</span>
          </div>

          <div className="inspector-body">
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', background: 'rgba(59, 130, 246, 0.08)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600, color: '#93c5fd', marginBottom: '2px' }}>
                <Info size={12} />
                <span>Explainable Retrieval</span>
              </div>
              Inspect how semantic vectors (Dense) and keyword matching (Sparse BM25) combine via Reciprocal Rank Fusion.
            </div>

            {latestChunks.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '2rem 1rem', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                Run a query to inspect live Dense vs. Sparse ranking metrics.
              </div>
            ) : (
              latestChunks.map((chunk, cIdx) => (
                <div key={cIdx} className="rank-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, color: '#f3f4f6' }}>
                      Rank #{cIdx + 1}
                    </span>
                    <span className="code-font" style={{ color: '#34d399', fontWeight: 600 }}>
                      RRF: {chunk.rrf_score?.toFixed(5) || 'N/A'}
                    </span>
                  </div>

                  <div style={{ color: '#9ca3af', fontSize: '0.75rem' }}>
                    {chunk.file_name} • Page {chunk.page}
                  </div>

                  {/* Rank metrics breakdown */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginTop: '4px' }}>
                    <div style={{ background: 'rgba(59, 130, 246, 0.1)', padding: '4px 6px', borderRadius: '4px' }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>Dense Rank:</span>
                      <div className="code-font" style={{ color: '#60a5fa', fontWeight: 600 }}>
                        {chunk.dense_rank ? `#${chunk.dense_rank}` : 'Unranked'}
                      </div>
                    </div>
                    <div style={{ background: 'rgba(139, 92, 246, 0.1)', padding: '4px 6px', borderRadius: '4px' }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>Sparse Rank:</span>
                      <div className="code-font" style={{ color: '#a78bfa', fontWeight: 600 }}>
                        {chunk.sparse_rank ? `#${chunk.sparse_rank}` : 'Unranked'}
                      </div>
                    </div>
                  </div>

                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px', maxHeight: '48px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    "{chunk.text}"
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
