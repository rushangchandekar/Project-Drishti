import React, { useState } from 'react';
import {
    MapPin, Clock, Radio, RefreshCw, AlertTriangle, Edit2, Check, X
} from 'lucide-react';
import { motion } from 'framer-motion';

function TopBar({ status, liveTime, formatTime }) {
    const [venueName, setVenueName] = useState("Main Venue - Section Alpha");
    const [isEditing, setIsEditing] = useState(false);
    const [tempName, setTempName] = useState(venueName);

    const handleEditClick = () => {
        setTempName(venueName);
        setIsEditing(true);
    };

    const handleSave = () => {
        setVenueName(tempName);
        setIsEditing(false);
    };

    const handleCancel = () => {
        setIsEditing(false);
    };

    return (
        <header className="topbar">
            <div className="topbar-left">
                <h1 className="project-title">
                    <span className="gradient-text">DRISHTI</span>
                    <span className="subtitle">Crowd Intelligence System</span>
                </h1>
            </div>

            <div className="topbar-center">
                <div className="location-badge">
                    <MapPin size={16} />
                    {isEditing ? (
                        <div className="edit-venue-container">
                            <input
                                type="text"
                                value={tempName}
                                onChange={(e) => setTempName(e.target.value)}
                                className="venue-input"
                                autoFocus
                            />
                            <button onClick={handleSave} className="edit-btn save"><Check size={14} /></button>
                            <button onClick={handleCancel} className="edit-btn cancel"><X size={14} /></button>
                        </div>
                    ) : (
                        <div className="venue-display" onClick={handleEditClick} title="Click to edit">
                            <span>{venueName}</span>
                            <Edit2 size={12} className="edit-icon" />
                        </div>
                    )}
                </div>

                <div className="live-badge">
                    <div className="live-dot" />
                    <span>LIVE</span>
                    <Clock size={14} />
                    <span className="live-time">{formatTime(liveTime)}</span>
                </div>

                <div className="status-badge" data-status={status.connected ? 'online' : 'offline'}>
                    <Radio size={14} />
                    <span>{status.connected ? 'System Online' : 'Disconnected'}</span>
                </div>
            </div>

            <div className="topbar-right">
                <button className="icon-btn" title="Refresh" onClick={() => window.location.reload()}>
                    <RefreshCw size={20} />
                </button>

                <motion.button
                    className="btn btn-danger manual-override"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <AlertTriangle size={18} />
                    Manual Override
                </motion.button>
            </div>

            <style>{`
        .topbar {
          height: var(--topbar-height);
          background: var(--bg-secondary);
          border-bottom: 1px solid var(--border-primary);
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 24px;
        }

        .project-title {
          display: flex;
          flex-direction: column;
        }

        .project-title .gradient-text {
          font-size: 22px;
          font-weight: 800;
          letter-spacing: 2px;
        }

        .project-title .subtitle {
          font-size: 11px;
          color: var(--text-secondary);
          font-weight: 500;
          letter-spacing: 0.5px;
        }

        .topbar-center {
          display: flex;
          align-items: center;
          gap: 20px;
        }

        .location-badge {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          background: var(--bg-tertiary);
          border-radius: 8px;
          font-size: 13px;
          color: var(--text-secondary);
          min-width: 200px;
          justify-content: center;
        }

        .venue-display {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }

        .venue-display:hover .edit-icon {
          opacity: 1;
        }

        .edit-icon {
          opacity: 0.5;
          transition: opacity 0.2s;
        }

        .edit-venue-container {
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .venue-input {
          background: var(--bg-primary);
          border: 1px solid var(--border-primary);
          color: var(--text-primary);
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 12px;
          width: 150px;
        }

        .edit-btn {
          background: transparent;
          border: none;
          cursor: pointer;
          color: var(--text-secondary);
          padding: 2px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .edit-btn.save:hover { color: var(--status-safe); }
        .edit-btn.cancel:hover { color: var(--status-critical); }

        .live-badge {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          background: rgba(63, 185, 80, 0.15);
          border: 1px solid rgba(63, 185, 80, 0.3);
          border-radius: 8px;
          font-size: 13px;
          font-weight: 600;
          color: var(--status-safe);
        }

        .live-dot {
          width: 8px;
          height: 8px;
          background: var(--status-safe);
          border-radius: 50%;
          animation: pulse 1.5s ease-in-out infinite;
        }

        .live-time {
          font-family: 'SF Mono', 'Consolas', monospace;
          font-size: 12px;
        }

        .status-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 14px;
          background: var(--bg-tertiary);
          border-radius: 8px;
          font-size: 12px;
          color: var(--text-secondary);
        }

        .status-badge[data-status="online"] { color: var(--status-safe); }
        .status-badge[data-status="offline"] { color: var(--status-critical); }

        .topbar-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .icon-btn {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          border: 1px solid var(--border-primary);
          background: var(--bg-tertiary);
          color: var(--text-secondary);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
        }

        .icon-btn:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
          border-color: var(--accent-blue);
        }

        .manual-override {
          font-size: 13px;
          padding: 10px 18px;
        }
      `}</style>
        </header>
    );
}

export default TopBar;
