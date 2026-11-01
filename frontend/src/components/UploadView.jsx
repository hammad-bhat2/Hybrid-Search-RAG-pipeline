import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, FileText, Trash2, RefreshCw, CheckCircle, AlertCircle, Layers, BookOpen } from 'lucide-react';

export default function UploadView({ onDocumentUploaded }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/v1/documents');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Error fetching documents:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleUpload = async (file) => {
    if (!file) return;

    setUploading(true);
    setUploadStatus({ type: 'info', message: `Uploading and indexing '${file.name}' (Parsing, embedding & indexing)...` });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/v1/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok && data.status === 'success') {
        setUploadStatus({
          type: 'success',
          message: `Indexed '${data.file_name}' (${data.chunks_indexed} chunks, ${data.total_pages} pages).`
        });
        fetchDocuments();
        if (onDocumentUploaded) onDocumentUploaded();
      } else {
        setUploadStatus({
          type: 'error',
          message: data.detail || data.message || 'Upload failed.'
        });
      }
    } catch (err) {
      setUploadStatus({ type: 'error', message: `Upload error: ${err.message}` });
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (fileName) => {
    if (!window.confirm(`Are you sure you want to delete '${fileName}' and all its indexed chunks?`)) {
      return;
    }

    try {
      const res = await fetch(`/api/v1/documents/${encodeURIComponent(fileName)}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        fetchDocuments();
        if (onDocumentUploaded) onDocumentUploaded();
      }
    } catch (err) {
      alert(`Delete error: ${err.message}`);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Upload Dropzone Card */}
      <div className="glass-card" style={{ padding: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          Document Ingestion & Indexing
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
          Upload PDF, DOCX, TXT, Markdown, or CSV files. The pipeline extracts text, builds overlapping chunks, generates dense vector (bge-small) & sparse BM25 embeddings, and stores them in Qdrant.
        </p>

        <div
          className={`dropzone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.doc,.txt,.md,.markdown,.csv"
            style={{ display: 'none' }}
            onChange={(e) => e.target.files && handleUpload(e.target.files[0])}
            disabled={uploading}
          />
          <UploadCloud size={44} color="#3b82f6" style={{ margin: '0 auto 1rem' }} />
          <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.35rem' }}>
            {uploading ? 'Processing & Indexing Document...' : 'Click to upload or drag & drop documents'}
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            Supports PDF, Word (.docx), Markdown (.md), Plain Text (.txt), CSV
          </p>
        </div>

        {uploadStatus && (
          <div style={{
            marginTop: '1.25rem',
            padding: '0.85rem 1.25rem',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            fontSize: '0.875rem',
            background: uploadStatus.type === 'success' ? 'rgba(16, 185, 129, 0.15)' :
                        uploadStatus.type === 'error' ? 'rgba(244, 63, 94, 0.15)' : 'rgba(59, 130, 246, 0.15)',
            border: `1px solid ${uploadStatus.type === 'success' ? 'rgba(16, 185, 129, 0.3)' :
                                 uploadStatus.type === 'error' ? 'rgba(244, 63, 94, 0.3)' : 'rgba(59, 130, 246, 0.3)'}`,
            color: uploadStatus.type === 'success' ? '#6ee7b7' :
                   uploadStatus.type === 'error' ? '#fda4af' : '#93c5fd'
          }}>
            {uploadStatus.type === 'success' && <CheckCircle size={18} />}
            {uploadStatus.type === 'error' && <AlertCircle size={18} />}
            {uploadStatus.type === 'info' && <RefreshCw size={18} className="spin" />}
            <span>{uploadStatus.message}</span>
          </div>
        )}
      </div>

      {/* Indexed Documents Table */}
      <div className="glass-card">
        <div style={{
          padding: '1.25rem 1.5rem',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Indexed Knowledge Base</h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {documents.length} document{documents.length === 1 ? '' : 's'} indexed in Qdrant
            </span>
          </div>
          <button className="btn btn-secondary" onClick={fetchDocuments} disabled={loading}>
            <RefreshCw size={14} />
            Refresh
          </button>
        </div>

        {documents.length === 0 ? (
          <div style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <FileText size={40} style={{ margin: '0 auto 1rem', opacity: 0.4 }} />
            <p style={{ fontWeight: 500 }}>No documents indexed yet.</p>
            <p style={{ fontSize: '0.8rem', marginTop: '0.25rem' }}>Upload a file above to begin querying with hybrid search.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Document Name</th>
                  <th>Pages</th>
                  <th>Indexed Chunks</th>
                  <th>Total Words</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.file_name}>
                    <td style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <FileText size={16} color="#60a5fa" />
                      {doc.file_name}
                    </td>
                    <td>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                        <BookOpen size={13} color="var(--text-dim)" />
                        {doc.total_pages}
                      </span>
                    </td>
                    <td>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: 'var(--radius-sm)',
                        background: 'rgba(59, 130, 246, 0.15)',
                        color: '#93c5fd',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}>
                        <Layers size={12} />
                        {doc.total_chunks}
                      </span>
                    </td>
                    <td style={{ color: 'var(--text-muted)' }}>{doc.total_words.toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-danger"
                        style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
                        onClick={() => handleDelete(doc.file_name)}
                      >
                        <Trash2 size={13} />
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
