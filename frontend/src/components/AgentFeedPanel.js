import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Bot, ShieldAlert, Cpu } from 'lucide-react';

function AgentFeedPanel({ agentFeed }) {

    // Pick an icon based on the agent name
    const getAgentIcon = (agentName) => {
        if (!agentName) return <Bot size={18} />;
        const name = agentName.toLowerCase();
        if (name.includes('security') || name.includes('anomaly')) return <ShieldAlert size={18} />;
        if (name.includes('fire') || name.includes('medical') || name.includes('evacuation')) return <Activity size={18} />;
        return <Cpu size={18} />;
    };

    return (
        <aside className="agent-feed-panel">
            <div className="panel-header">
                <h2>
                    <Bot size={18} />
                    Live Agent Activity
                </h2>
                <span className="feed-count">{agentFeed.length}</span>
            </div>

            <div className="feed-list">
                <AnimatePresence>
                    {agentFeed.length === 0 ? (
                        <div className="no-feed">
                            <Activity size={40} className="pulse-icon" />
                            <p>Awaiting Agent Actions</p>
                            <span>Monitoring events in real-time</span>
                        </div>
                    ) : (
                        agentFeed.map((action, idx) => (
                            <motion.div
                                key={action.id || idx}
                                className="feed-card"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                            >
                                <div className="feed-icon">
                                    {getAgentIcon(action.agent)}
                                </div>
                                <div className="feed-content">
                                    <div className="feed-header-line">
                                        <h4>{action.agent || "System Agent"}</h4>
                                        <span className="feed-time">
                                            {action.timestamp ? new Date(action.timestamp).toLocaleTimeString() : ''}
                                        </span>
                                    </div>
                                    <div className="feed-status-badge">
                                        {action.status || 'EXECUTED'}
                                    </div>
                                    <div className="feed-details">
                                        {action.data && action.data.actions_taken && action.data.actions_taken.length > 0 ? (
                                            <ul>
                                                {action.data.actions_taken.slice(0, 2).map((a, i) => (
                                                    <li key={i}>
                                                        <span className="action-dot"></span>
                                                        {a.action}: {a.status || 'DONE'}
                                                    </li>
                                                ))}
                                                {action.data.actions_taken.length > 2 && (
                                                    <li className="more-actions">+{action.data.actions_taken.length - 2} more...</li>
                                                )}
                                            </ul>
                                        ) : action.data && action.data.summary ? (
                                            <p className="summary-text">{action.data.summary.substring(0, 80)}...</p>
                                        ) : (
                                            <p>Action completed successfully.</p>
                                        )}
                                    </div>
                                </div>
                            </motion.div>
                        ))
                    )}
                </AnimatePresence>
            </div>

            <style>{`
                .agent-feed-panel {
                    width: 340px;
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
                    background: linear-gradient(90deg, rgba(88,101,242,0.1) 0%, transparent 100%);
                }

                .panel-header h2 {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 16px;
                    font-weight: 600;
                    color: var(--accent-purple);
                }

                .feed-count {
                    background: var(--accent-purple);
                    color: white;
                    font-size: 12px;
                    font-weight: 600;
                    padding: 4px 10px;
                    border-radius: 12px;
                }

                .feed-list {
                    flex: 1;
                    overflow-y: auto;
                    padding: 16px;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }

                .no-feed {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    color: var(--text-muted);
                    gap: 8px;
                    padding: 40px 0;
                }

                .pulse-icon {
                    animation: pulse 2s infinite;
                    color: var(--accent-purple);
                    opacity: 0.5;
                }

                @keyframes pulse {
                    0% { transform: scale(1); opacity: 0.5; }
                    50% { transform: scale(1.1); opacity: 1; }
                    100% { transform: scale(1); opacity: 0.5; }
                }

                .feed-card {
                    background: var(--bg-tertiary);
                    border-radius: 12px;
                    padding: 14px;
                    display: flex;
                    gap: 12px;
                    border-left: 3px solid var(--accent-purple);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }

                .feed-icon {
                    flex-shrink: 0;
                    color: var(--accent-purple);
                    background: rgba(163, 113, 247, 0.1);
                    padding: 8px;
                    border-radius: 8px;
                    height: fit-content;
                }

                .feed-content {
                    flex: 1;
                    overflow: hidden;
                }

                .feed-header-line {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 6px;
                }

                .feed-header-line h4 {
                    font-size: 13px;
                    font-weight: 600;
                    color: var(--text-primary);
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }

                .feed-time {
                    font-size: 10px;
                    color: var(--text-muted);
                }

                .feed-status-badge {
                    display: inline-block;
                    font-size: 10px;
                    font-weight: 600;
                    padding: 2px 6px;
                    border-radius: 4px;
                    background: rgba(63, 185, 80, 0.15);
                    color: var(--status-safe);
                    margin-bottom: 8px;
                    letter-spacing: 0.5px;
                }

                .feed-details {
                    font-size: 12px;
                    color: var(--text-secondary);
                }

                .feed-details ul {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }

                .feed-details li {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                }

                .action-dot {
                    width: 4px;
                    height: 4px;
                    background: var(--accent-purple);
                    border-radius: 50%;
                }

                .more-actions {
                    font-size: 11px;
                    color: var(--text-muted);
                    font-style: italic;
                    margin-top: 2px;
                }
                
                .summary-text {
                    line-height: 1.4;
                    color: var(--text-secondary);
                }
            `}</style>
        </aside>
    );
}

export default AgentFeedPanel;
