import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
    Users, Activity, Zap, Send, X, Maximize2,
    Minimize2, HelpCircle, Sparkles
} from 'lucide-react';

function GlassmorphismChat({ isOpen, setIsOpen, isExpanded, setIsExpanded, messages, input, setInput, onSend, isLoading, status }) {
    const messagesEndRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    if (!isOpen) return null;

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSend();
        }
    };

    return (
        <motion.div
            className={`glass-chat ${isExpanded ? 'expanded' : ''}`}
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
        >
            <div className="chat-header">
                <div className="chat-branding">
                    <div className="gemini-logo">
                        <Sparkles size={18} />
                    </div>
                    <div className="chat-title">
                        <span className="title-main">Gemini LLM Agent</span>
                        <span className="title-sub">Powered by AI</span>
                    </div>
                </div>
                <div className="chat-controls">
                    <button className="help-badge">
                        <HelpCircle size={12} />
                        Help
                    </button>
                    <button className="control-btn" onClick={() => setIsExpanded(!isExpanded)}>
                        {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                    </button>
                    <button className="control-btn close" onClick={() => setIsOpen(false)}>
                        <X size={16} />
                    </button>
                </div>
            </div>

            <div className="context-bar">
                <div className="context-item">
                    <Users size={12} />
                    <span>{status.person_count} people</span>
                </div>
                <div className="context-item">
                    <Activity size={12} />
                    <span>{status.density_level}</span>
                </div>
                <div className="context-item">
                    <Zap size={12} />
                    <span>Risk: {status.risk_score}</span>
                </div>
            </div>

            <div className="chat-messages">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.type}`}>
                        {msg.type === 'user' ? (
                            <>
                                <div className="message-content user-content">
                                    {msg.text}
                                </div>
                                <div className="user-avatar">{msg.initials}</div>
                            </>
                        ) : (
                            <>
                                <div className="bot-avatar">
                                    <Sparkles size={14} />
                                </div>
                                <div className="message-content bot-content">
                                    <div className="message-text" style={{ color: msg.isError ? 'var(--status-critical)' : 'inherit' }}>
                                        {msg.text}
                                    </div>
                                    {msg.agents && (
                                        <div className="agent-attribution">
                                            {msg.agents.map((agent, i) => (
                                                <span key={i} className="agent-tag">{agent}</span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </>
                        )}
                    </div>
                ))}

                {isLoading && (
                    <div className="message bot">
                        <div className="bot-avatar">
                            <Sparkles size={14} />
                        </div>
                        <div className="message-content bot-content">
                            <div className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            <div className="quick-actions">
                <button onClick={() => setInput('How many people are in the venue?')}>Crowd count</button>
                <button onClick={() => setInput('What is the current risk level?')}>Risk analysis</button>
                <button onClick={() => setInput('Are there any active alerts?')}>Check alerts</button>
            </div>

            <div className="chat-input-area">
                <input
                    type="text"
                    placeholder="Ask a question..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    disabled={isLoading}
                />
                <button onClick={onSend} disabled={isLoading || !input.trim()}>
                    <Send size={18} />
                </button>
            </div>

            <style>{`
        .glass-chat {
          position: fixed;
          bottom: 120px;
          right: 24px;
          width: 400px;
          height: 520px;
          background: rgba(13, 17, 23, 0.85);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 20px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          z-index: 2000;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .glass-chat.expanded {
          width: 500px;
          height: 650px;
        }

        .chat-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 20px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }

        .chat-branding {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .gemini-logo {
          width: 36px;
          height: 36px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }

        .chat-title {
          display: flex;
          flex-direction: column;
        }

        .title-main {
          font-size: 14px;
          font-weight: 600;
          color: #a5b4fc;
        }

        .title-sub {
          font-size: 10px;
          color: var(--text-muted);
        }

        .chat-controls {
          display: flex;
          gap: 8px;
        }

        .help-badge {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 6px 12px;
          background: rgba(163, 113, 247, 0.15);
          border: none;
          border-radius: 6px;
          color: var(--accent-purple);
          font-size: 11px;
          cursor: pointer;
        }

        .control-btn {
          width: 32px;
          height: 32px;
          border-radius: 8px;
          border: none;
          background: rgba(255, 255, 255, 0.05);
          color: var(--text-secondary);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .control-btn:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .context-bar {
          display: flex;
          gap: 16px;
          padding: 10px 20px;
          background: rgba(255, 255, 255, 0.02);
          border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .context-item {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          color: var(--text-muted);
        }

        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .message {
          display: flex;
          gap: 12px;
        }

        .message.user {
          flex-direction: row-reverse;
        }

        .user-avatar {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: 600;
          color: white;
        }

        .bot-avatar {
          width: 32px;
          height: 32px;
          border-radius: 10px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }

        .message-content {
          max-width: 80%;
        }

        .user-content {
          background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
          color: white;
          padding: 12px 16px;
          border-radius: 16px 16px 4px 16px;
          font-size: 13px;
        }

        .bot-content {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          padding: 14px 16px;
          border-radius: 16px 16px 16px 4px;
        }

        .message-text {
          font-size: 13px;
          line-height: 1.6;
          color: var(--text-primary);
        }

        .agent-attribution {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-top: 10px;
          padding-top: 10px;
          border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .agent-tag {
          font-size: 10px;
          padding: 3px 8px;
          background: rgba(163, 113, 247, 0.15);
          color: var(--accent-purple);
          border-radius: 4px;
        }

        .typing-indicator {
          display: flex;
          gap: 4px;
          padding: 8px 0;
        }

        .typing-indicator span {
          width: 8px;
          height: 8px;
          background: var(--text-muted);
          border-radius: 50%;
          animation: typing 1.4s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30% { transform: translateY(-6px); opacity: 1; }
        }

        .quick-actions {
          display: flex;
          gap: 8px;
          padding: 12px 20px;
          border-top: 1px solid rgba(255, 255, 255, 0.04);
          overflow-x: auto;
        }

        .quick-actions button {
          padding: 8px 14px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 20px;
          color: var(--text-secondary);
          font-size: 11px;
          cursor: pointer;
          white-space: nowrap;
        }

        .quick-actions button:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: var(--accent-blue);
        }

        .chat-input-area {
          display: flex;
          gap: 10px;
          padding: 16px 20px;
          border-top: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(0, 0, 0, 0.2);
        }

        .chat-input-area input {
          flex: 1;
          padding: 14px 18px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(255, 255, 255, 0.05);
          color: var(--text-primary);
          font-size: 13px;
          outline: none;
        }

        .chat-input-area input:focus {
          border-color: var(--accent-blue);
        }

        .chat-input-area button {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          border: none;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .chat-input-area button:hover:not(:disabled) {
          transform: scale(1.05);
        }

        .chat-input-area button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
        </motion.div>
    );
}

export default GlassmorphismChat;
