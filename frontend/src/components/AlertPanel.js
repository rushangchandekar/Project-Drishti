import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bell, AlertTriangle, AlertCircle, CheckCircle } from 'lucide-react';

function AlertPanel({ alerts, alertFilter, setAlertFilter }) {
    return (
        <aside className="alert-panel">
            <div className="panel-header">
                <h2>
                    <Bell size={18} />
                    Real-Time Alerts
                </h2>
                <span className="alert-count">{alerts.length}</span>
            </div>

            <div className="filter-tabs">
                {['all', 'critical', 'warning'].map(filter => (
                    <button
                        key={filter}
                        className={`filter-tab ${alertFilter === filter ? 'active' : ''}`}
                        onClick={() => setAlertFilter(filter)}
                    >
                        {filter.charAt(0).toUpperCase() + filter.slice(1)}
                    </button>
                ))}
            </div>

            <div className="alerts-list">
                <AnimatePresence>
                    {alerts.length === 0 ? (
                        <div className="no-alerts">
                            <CheckCircle size={40} />
                            <p>No active alerts</p>
                            <span>System operating normally</span>
                        </div>
                    ) : (
                        alerts.map(alert => (
                            <motion.div
                                key={alert.id}
                                className={`alert-card ${alert.type}`}
                                initial={{ opacity: 0, x: 50 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -50 }}
                            >
                                <div className="alert-icon">
                                    {alert.type === 'critical' ? <AlertCircle size={20} /> : <AlertTriangle size={20} />}
                                </div>
                                <div className="alert-content">
                                    <h4>{alert.title}</h4>
                                    <p>{alert.message}</p>
                                    <div className="alert-meta">
                                        <span className="alert-time">{alert.time}</span>
                                        <span className="alert-agent">{alert.agent}</span>
                                    </div>
                                </div>
                            </motion.div>
                        ))
                    )}
                </AnimatePresence>
            </div>

            <style>{`
        .alert-panel {
          width: var(--right-panel-width);
          background: var(--bg-secondary);
          border-radius: 16px;
          border: 1px solid var(--border-primary);
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .panel-header {
          padding: 20px;
          border-bottom: 1px solid var(--border-primary);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .panel-header h2 {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 16px;
          font-weight: 600;
        }

        .alert-count {
          background: var(--accent-blue);
          color: white;
          font-size: 12px;
          font-weight: 600;
          padding: 4px 10px;
          border-radius: 12px;
        }

        .filter-tabs {
          display: flex;
          padding: 12px 16px;
          gap: 8px;
          border-bottom: 1px solid var(--border-primary);
        }

        .filter-tab {
          flex: 1;
          padding: 8px 12px;
          border: none;
          border-radius: 8px;
          background: var(--bg-tertiary);
          color: var(--text-secondary);
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .filter-tab:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
        }

        .filter-tab.active {
          background: var(--accent-blue);
          color: white;
        }

        .alerts-list {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .no-alerts {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          color: var(--text-muted);
          gap: 8px;
        }

        .alert-card {
          background: var(--bg-tertiary);
          border-radius: 12px;
          padding: 16px;
          display: flex;
          gap: 14px;
          border-left: 4px solid;
        }

        .alert-card.critical {
          border-color: var(--status-critical);
          background: rgba(248, 81, 73, 0.08);
        }

        .alert-card.warning {
          border-color: var(--status-warning);
          background: rgba(210, 153, 34, 0.08);
        }

        .alert-icon {
          flex-shrink: 0;
        }

        .alert-card.critical .alert-icon {
          color: var(--status-critical);
        }

        .alert-card.warning .alert-icon {
          color: var(--status-warning);
        }

        .alert-content {
          flex: 1;
        }

        .alert-content h4 {
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 4px;
        }

        .alert-content p {
          font-size: 12px;
          color: var(--text-secondary);
          margin-bottom: 8px;
        }

        .alert-meta {
          display: flex;
          gap: 12px;
          font-size: 11px;
        }

        .alert-time {
          color: var(--text-muted);
        }

        .alert-agent {
          color: var(--accent-purple);
          background: rgba(163, 113, 247, 0.15);
          padding: 2px 8px;
          border-radius: 4px;
        }
      `}</style>
        </aside>
    );
}

export default AlertPanel;
