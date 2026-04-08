'use client';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Minus, Maximize2, Minimize2, X, Send } from 'lucide-react';

export default function GeminiChat() {
  const [isOpen, setIsOpen] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    const question = input;
    setMessages(prev => [...prev, { role: 'user', text: question }]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'bot', text: data.answer || 'No response received' }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'bot', text: 'Error connecting to backend.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) {
    return (
      <motion.button
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        onClick={() => setIsOpen(true)}
        style={{
          position: 'absolute',
          bottom: '24px',
          right: '24px',
          background: 'var(--gradient-brand)',
          color: 'white',
          border: 'none',
          padding: '12px 24px',
          borderRadius: '50px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          cursor: 'pointer',
          zIndex: 50,
          boxShadow: 'var(--shadow-lg)'
        }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <Sparkles size={18} />
        <span style={{ fontWeight: 600, fontSize: '14px' }}>AI Assistant</span>
      </motion.button>
    );
  }

  return (
    <motion.div
      initial={{ y: 50, opacity: 0 }}
      animate={{ 
        y: 0, 
        opacity: 1, 
        width: isExpanded && !isMinimized ? '600px' : '380px', 
        height: isMinimized ? 'auto' : (isExpanded ? '600px' : '450px') 
      }}
      transition={{ type: "spring", bounce: 0, duration: 0.4 }}
      style={{
        position: 'absolute',
        bottom: '24px',
        right: '24px',
        background: 'rgba(12, 16, 24, 0.85)',
        backdropFilter: 'blur(16px)',
        border: '1px solid var(--accent-blue)',
        borderRadius: '16px',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: 'var(--shadow-glow-blue)',
        zIndex: 50
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', borderBottom: '1px solid var(--border-primary)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, fontSize: '13px' }}>
          <Sparkles size={16} color="var(--accent-blue)" />
          AI Assistant
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={() => setIsMinimized(!isMinimized)} style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }} title="Minimize">
            <Minus size={14} />
          </button>
          {!isMinimized && (
            <button onClick={() => setIsExpanded(!isExpanded)} style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }} title="Maximize/Restore">
              {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={12} />}
            </button>
          )}
          <button onClick={() => setIsOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }} title="Close">
            <X size={16} />
          </button>
        </div>
      </div>

      <AnimatePresence>
        {!isMinimized && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
          >
            {/* Messages */}
            <div style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
              {messages.length === 0 && (
                 <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px', marginTop: '20px' }}>
                    Agent ready. Ask a question about the venue or current risks.
                 </div>
              )}
              {messages.map((m, i) => (
                <div key={i} style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                  {m.role === 'user' ? (
                    <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--bg-elevated)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', color: '#a78bfa', fontWeight: 600, border: '1px solid rgba(167, 139, 250, 0.3)', flexShrink: 0 }}>U</div>
                  ) : (
                    <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'linear-gradient(135deg, #d29922 0%, #a371f7 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', flexShrink: 0 }}>
                      <Sparkles size={14} />
                    </div>
                  )}
                  
                  <div style={{ flex: 1, background: m.role === 'user' ? 'rgba(255,255,255,0.05)' : 'transparent', padding: m.role === 'user' ? '8px 12px' : '0', borderRadius: '8px', fontSize: '13px', color: 'var(--text-primary)', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
                    {m.role === 'bot' && (
                      <div style={{ color: 'var(--accent-blue)', fontWeight: 600, marginBottom: '4px' }}>Gemini LLM Agent:</div>
                    )}
                    {m.text}
                  </div>
                </div>
              ))}
              {isLoading && (
                 <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                    <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'linear-gradient(135deg, #d29922 0%, #a371f7 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', flexShrink: 0 }}>
                      <Sparkles size={14} />
                    </div>
                    <div style={{ flex: 1, color: 'var(--text-muted)', fontSize: '13px', display: 'flex', alignItems: 'center', padding: '4px 0' }}>
                       <span className="animate-pulse">Analyzing...</span>
                    </div>
                 </div>
              )}
            </div>

            {/* Input */}
            <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border-primary)', display: 'flex', gap: '8px' }}>
              <input 
                type="text" 
                placeholder="Ask a question..." 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                disabled={isLoading}
                style={{ flex: 1, background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-primary)', borderRadius: '24px', padding: '8px 16px', color: 'white', fontSize: '13px', outline: 'none', opacity: isLoading ? 0.5 : 1 }}
              />
              <button 
                onClick={handleSend} 
                disabled={isLoading || !input.trim()}
                style={{ width: '36px', height: '36px', borderRadius: '50%', background: 'transparent', border: 'none', color: isLoading || !input.trim() ? 'var(--text-muted)' : 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: isLoading || !input.trim() ? 'default' : 'pointer' }}>
                <Send size={16} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
