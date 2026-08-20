'use client';
import { useState } from 'react';
import { 
  Flame, 
  Users, 
  LogOut, 
  AlertTriangle, 
  TrendingUp, 
  HeartPulse, 
  Radio, 
  Brain, 
  Shield, 
  BrainCircuit, 
  Activity, 
  Network, 
  CheckCircle2, 
  AlertCircle, 
  Clock, 
  Terminal, 
  Play, 
  ChevronDown, 
  ChevronUp 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const defaultAgents = {
  FireAgent: { name: "Fire Agent", category: "emergency", icon: "flame", description: "Handles fire detection, sprinkler activation, fire station contact" },
  CrowdAgent: { name: "Crowd Agent", category: "emergency", icon: "users", description: "Manages crowd density, gate control, crowd dispersal" },
  EvacAgent: { name: "Evac Agent", category: "emergency", icon: "log-out", description: "Coordinates evacuation routes, emergency exits, PA announcements" },
  AnomalyAgent: { name: "Anomaly Agent", category: "intelligence", icon: "alert-triangle", description: "Investigates unusual crowd behavior, suspicious activity" },
  ForecastAgent: { name: "Forecast Agent", category: "intelligence", icon: "trending-up", description: "Predicts crowd trends, escalation risk, resource needs" },
  MedicAgent: { name: "Medic Agent", category: "intelligence", icon: "heart-pulse", description: "Deploys medical teams for stampede, crush, or health emergencies" },
  DispatchAgent: { name: "Dispatch Agent", category: "operations", icon: "radio", description: "Dispatches security, staff, or emergency services to zones" },
  LLMAgent: { name: "LLM Agent", category: "operations", icon: "brain", description: "Generates situation summaries, incident reports, command briefs" },
  SecurityAgent: { name: "Security Agent", category: "operations", icon: "shield", description: "Monitors perimeter, suspicious behavior, access control" },
};

const categoryConfig = {
  emergency: {
    label: "Emergency & Safety",
    color: "#ef4444",
    bg: "rgba(239, 68, 68, 0.05)",
    border: "rgba(239, 68, 68, 0.15)",
    badgeBg: "rgba(239, 68, 68, 0.1)",
  },
  intelligence: {
    label: "Cognitive Intelligence",
    color: "#f59e0b",
    bg: "rgba(245, 158, 11, 0.05)",
    border: "rgba(245, 158, 11, 0.15)",
    badgeBg: "rgba(245, 158, 11, 0.1)",
  },
  operations: {
    label: "Tactical Operations",
    color: "#3b82f6",
    bg: "rgba(59, 130, 246, 0.05)",
    border: "rgba(59, 130, 246, 0.15)",
    badgeBg: "rgba(59, 130, 246, 0.1)",
  }
};

const getAgentIcon = (iconName, color, size = 20) => {
  switch (iconName) {
    case 'flame': return <Flame size={size} color={color} />;
    case 'users': return <Users size={size} color={color} />;
    case 'log-out': return <LogOut size={size} color={color} />;
    case 'alert-triangle': return <AlertTriangle size={size} color={color} />;
    case 'trending-up': return <TrendingUp size={size} color={color} />;
    case 'heart-pulse': return <HeartPulse size={size} color={color} />;
    case 'radio': return <Radio size={size} color={color} />;
    case 'brain': return <Brain size={size} color={color} />;
    case 'shield': return <Shield size={size} color={color} />;
    default: return <BrainCircuit size={size} color={color} />;
  }
};

export default function AgentExecutionPanel({ agentStatuses = {}, agentFeed = [], liveStatus = {} }) {
  const [expandedTrace, setExpandedTrace] = useState(null);

  // Derive a live status message from the current detection frame for each agent
  const getLiveAgentMessage = (agentId) => {
    const s = liveStatus;
    if (!s || !s.connected) return null;

    switch (agentId) {
      case 'FireAgent':
        if (s.fire_detected)
          return `🔥 Fire detected (confidence ${Math.round((s.fire_confidence || 0) * 100)}%)`;
        return null;

      case 'CrowdAgent': {
        const count = s.person_count ?? 0;
        const density = s.density_level || '';
        if (['CRITICAL', 'VERY_HIGH', 'HIGH'].includes(density))
          return `${count} people detected — Density: ${density}`;
        if (count > 0)
          return `Monitoring ${count} people — ${density || 'NORMAL'}`;
        return null;
      }

      case 'EvacAgent':
        if (s.fire_detected) return 'Evacuation required due to fire';
        if (['CRITICAL', 'VERY_HIGH'].includes(s.density_level))
          return `Evacuation standby — ${s.density_level} density`;
        if (s.activities && s.activities.some(a => a.type === 'PANIC' || a.type === 'STAMPEDE'))
          return 'Stampede detected — PA system activated';
        return null;

      case 'AnomalyAgent':
        if (s.anomaly_detected && s.anomaly_type)
          return `Anomaly: ${s.anomaly_type}`;
        if (s.anomaly_detected)
          return 'Unusual crowd pattern detected';
        return null;

      case 'ForecastAgent': {
        const risk = Math.round(s.risk_score || 0);
        const trend = s.trend || 'STABLE';
        return `Risk ${risk}/100, trend ${trend}`;
      }

      case 'MedicAgent': {
        const medAct = s.activities && s.activities.find(a => a.type === 'FALL' || a.type === 'STAMPEDE');
        if (medAct) return medAct.description || `${medAct.type} detected`;
        return null;
      }

      case 'DispatchAgent': {
        const fightAct = s.activities && s.activities.find(a => a.type === 'FIGHT');
        if (fightAct) return fightAct.description || 'Fight/altercation detected';
        if (s.anomaly_severity === 'CRITICAL') return 'Critical anomaly — security dispatched';
        return null;
      }

      case 'LLMAgent':
        if (s.strategic_guidance && s.strategic_guidance !== 'NORMAL OPERATIONS: Continue routine surveillance.')
          return `High risk (${Math.round(s.risk_score || 0)}) — generate situation report`;
        return null;

      case 'SecurityAgent':
        if (s.anomaly_detected && s.anomaly_type)
          return `High-severity anomaly: ${s.anomaly_type}`;
        if ((s.risk_score || 0) > 75) return `High risk score (${Math.round(s.risk_score)}) — perimeter alert`;
        return null;

      default:
        return null;
    }
  };

  const normalizeCategory = (cat) => {
    if (!cat) return 'operations';
    const c = String(cat).toLowerCase();
    if (c.includes('emergency') || c.includes('safety')) return 'emergency';
    if (c.includes('cognitive') || c.includes('intelligence')) return 'intelligence';
    if (c.includes('tactical') || c.includes('operations')) return 'operations';
    return 'operations';
  };

  // Merge the polling/backend statuses with the default schema
  const mergedAgents = {};
  Object.entries(defaultAgents).forEach(([id, def]) => {
    const raw = agentStatuses[id] || {};
    const liveMessage = getLiveAgentMessage(id);
    mergedAgents[id] = {
      agent_id: id,
      status: raw.status || (raw.invocations > 0 ? 'completed' : 'idle'),
      invocation_count: raw.invocation_count ?? raw.invocations ?? 0,
      execution_time_ms: raw.execution_time_ms ?? (raw.latency ? parseFloat(raw.latency) : 0),
      last_invoked: raw.last_invoked || null,
      last_result: raw.last_result || null,
      last_error: raw.last_error || null,
      ...def,
      ...raw,
      // These always override the spread — live message takes priority over stale trigger_reason
      trigger_reason: liveMessage ?? raw.trigger_reason ?? null,
      is_live_message: liveMessage !== null,
      category: normalizeCategory(raw.category || def.category)
    };
  });


  const emergencyAgents = Object.values(mergedAgents).filter(a => a.category === 'emergency');
  const intelligenceAgents = Object.values(mergedAgents).filter(a => a.category === 'intelligence');
  const operationsAgents = Object.values(mergedAgents).filter(a => a.category === 'operations');

  const toggleTrace = (id) => {
    if (expandedTrace === id) {
      setExpandedTrace(null);
    } else {
      setExpandedTrace(id);
    }
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)', borderRadius: '16px', border: '1px solid var(--border-primary)', overflow: 'hidden' }}>
      
      {/* Header */}
      <div style={{ padding: '24px', borderBottom: '1px solid var(--border-primary)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(56, 189, 248, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <BrainCircuit size={24} color="#38bdf8" />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: '20px', fontWeight: 700, color: 'white', letterSpacing: '-0.3px' }}>Agent Orchestrator Dashboard</h2>
            <span style={{ fontSize: '13px', color: '#9ca3af' }}>Autonomous n8n Multi-Agent System Control Panel</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(255,255,255,0.03)', padding: '6px 12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', fontSize: '12px', color: '#cbd5e1' }}>
            <Activity size={14} color="#38bdf8" />
            <span>9 Actuators Loaded</span>
          </div>
        </div>
      </div>

      {/* Grid Container */}
      <div style={{ flex: 1, padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '28px' }}>
        
        {/* Category Columns / Rows */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
          
          {/* Emergency Category */}
          <CategorySection 
            cfg={categoryConfig.emergency} 
            agents={emergencyAgents} 
          />

          {/* Intelligence Category */}
          <CategorySection 
            cfg={categoryConfig.intelligence} 
            agents={intelligenceAgents} 
          />

          {/* Operations Category */}
          <CategorySection 
            cfg={categoryConfig.operations} 
            agents={operationsAgents} 
          />
          
        </div>

        {/* Divider */}
        <div style={{ height: '1px', backgroundColor: 'var(--border-primary)', margin: '8px 0' }} />

        {/* Action Feed Log */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Terminal size={18} color="#38bdf8" />
            <h3 style={{ fontSize: '14px', color: '#9ca3af', margin: 0, textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 600 }}>Action & Workflow Trace Log</h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <AnimatePresence initial={false}>
              {agentFeed.length === 0 ? (
                <div style={{ padding: '24px', textAlign: 'center', background: 'rgba(255,255,255,0.01)', borderRadius: '12px', border: '1px dashed rgba(255,255,255,0.05)', color: '#6b7280', fontSize: '13px' }}>
                  Awaiting agent execution triggers... Feed is currently empty.
                </div>
              ) : (
                agentFeed.slice(0, 8).map((event, idx) => (
                  <motion.div 
                    key={`${event.id || idx}-${event.agent || ''}-${idx}`}
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ 
                      display: 'flex', 
                      flexDirection: 'column', 
                      background: 'rgba(255,255,255,0.02)', 
                      padding: '16px', 
                      borderRadius: '12px', 
                      border: '1px solid rgba(255,255,255,0.04)',
                      transition: 'border-color 0.2s ease',
                      cursor: event.data ? 'pointer' : 'default'
                    }}
                    onClick={() => event.data && toggleTrace(event.id || idx)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                        <div style={{ width: '32px', height: '32px', borderRadius: '6px', background: event.status === 'FAILED' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(74, 222, 128, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          {event.status === 'FAILED' ? <AlertCircle size={16} color="#ef4444" /> : <CheckCircle2 size={16} color="#4ade80" />}
                        </div>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontWeight: 600, color: 'white', fontSize: '14px' }}>{event.agent || 'System Agent'}</span>
                            <span style={{ 
                              fontSize: '11px', 
                              padding: '1px 6px', 
                              borderRadius: '4px', 
                              background: event.status === 'FAILED' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(74, 222, 128, 0.15)', 
                              color: event.status === 'FAILED' ? '#ef4444' : '#4ade80',
                              fontWeight: 500
                            }}>
                              {event.status}
                            </span>
                          </div>
                          <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#9ca3af' }}>{event.trigger_reason || 'Periodic status check'}</p>
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '11px', color: '#6b7280' }}>{event.timestamp || event.time}</span>
                        {event.data && (
                          expandedTrace === (event.id || idx) ? <ChevronUp size={16} color="#6b7280" /> : <ChevronDown size={16} color="#6b7280" />
                        )}
                      </div>
                    </div>

                    {/* Expandable JSON Output */}
                    <AnimatePresence>
                      {event.data && expandedTrace === (event.id || idx) && (
                        <motion.div
                          initial={{ height: 0, opacity: 0, marginTop: 0 }}
                          animate={{ height: 'auto', opacity: 1, marginTop: 12 }}
                          exit={{ height: 0, opacity: 0, marginTop: 0 }}
                          style={{ overflow: 'hidden' }}
                          onClick={(e) => e.stopPropagation()} // Prevent closing when clicking JSON
                        >
                          <div style={{ background: '#0f172a', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)', fontFamily: 'monospace', fontSize: '12px', color: '#38bdf8', overflowX: 'auto', maxHeight: '180px' }}>
                            <pre style={{ margin: 0 }}>{JSON.stringify(event.data, null, 2)}</pre>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </div>

      </div>
    </div>
  );
}

function CategorySection({ cfg, agents }) {
  return (
    <div style={{ background: 'rgba(255, 255, 255, 0.01)', borderRadius: '12px', border: `1px solid ${cfg.border}`, padding: '18px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      
      {/* Category Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: cfg.color }} />
        <span style={{ fontSize: '13px', fontWeight: 700, color: cfg.color, textTransform: 'uppercase', letterSpacing: '0.8px' }}>{cfg.label}</span>
      </div>

      {/* Agents under this Category */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {agents.map((agent) => (
          <AgentCard key={agent.agent_id} agent={agent} cfg={cfg} />
        ))}
      </div>
    </div>
  );
}

function AgentCard({ agent, cfg }) {
  const isRunning = agent.status === 'running';
  const isCompleted = agent.status === 'completed';
  const isError = agent.status === 'error';

  let statusBg = 'rgba(255,255,255,0.02)';
  let statusBorder = 'rgba(255,255,255,0.05)';
  let statusIndicatorColor = '#9ca3af';

  if (isRunning) {
    statusBg = 'rgba(56, 189, 248, 0.04)';
    statusBorder = 'rgba(56, 189, 248, 0.2)';
    statusIndicatorColor = '#38bdf8';
  } else if (isCompleted) {
    statusBg = 'rgba(74, 222, 128, 0.03)';
    statusBorder = 'rgba(74, 222, 128, 0.15)';
    statusIndicatorColor = '#4ade80';
  } else if (isError) {
    statusBg = 'rgba(239, 68, 68, 0.03)';
    statusBorder = 'rgba(239, 68, 68, 0.15)';
    statusIndicatorColor = '#ef4444';
  }

  return (
    <div style={{ 
      background: statusBg, 
      border: `1px solid ${statusBorder}`, 
      borderRadius: '10px', 
      padding: '14px', 
      display: 'flex', 
      flexDirection: 'column', 
      gap: '8px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      
      {/* Top Details */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        
        {/* Left Side Icon + Name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ 
            width: '34px', 
            height: '34px', 
            borderRadius: '6px', 
            background: isRunning ? 'rgba(56, 189, 248, 0.12)' : 'rgba(255,255,255,0.03)', 
            display: 'flex', 
            alignItems: 'center', 
            justify: 'center',
            justifyContent: 'center',
            border: isRunning ? '1px solid rgba(56, 189, 248, 0.25)' : '1px solid rgba(255, 255, 255, 0.05)'
          }}>
            {getAgentIcon(agent.icon, isRunning ? '#38bdf8' : cfg.color)}
          </div>
          <div>
            <h4 style={{ margin: 0, color: 'white', fontSize: '14px', fontWeight: 600 }}>{agent.name}</h4>
            <span style={{ color: '#6b7280', fontSize: '11px' }}>{agent.agent_id}</span>
          </div>
        </div>

        {/* Right Side Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {isRunning ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div 
                style={{ 
                  width: '6px', 
                  height: '6px', 
                  borderRadius: '50%', 
                  backgroundColor: '#38bdf8',
                  boxShadow: '0 0 8px #38bdf8',
                  animation: 'pulse 1.5s infinite'
                }} 
              />
              <span style={{ fontSize: '11px', color: '#38bdf8', fontWeight: 700, textTransform: 'uppercase' }}>Running</span>
            </div>
          ) : isCompleted ? (
            <span style={{ fontSize: '10px', color: '#4ade80', background: 'rgba(74, 222, 128, 0.08)', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(74, 222, 128, 0.15)', fontWeight: 600 }}>
              {agent.execution_time_ms ? `${parseFloat(agent.execution_time_ms).toFixed(1)}ms` : 'Ready'}
            </span>
          ) : isError ? (
            <span style={{ fontSize: '10px', color: '#ef4444', background: 'rgba(239, 68, 68, 0.08)', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(239, 68, 68, 0.15)', fontWeight: 600 }}>
              ERR
            </span>
          ) : (
            <span style={{ fontSize: '10px', color: '#6b7280', background: 'rgba(255, 255, 255, 0.02)', padding: '2px 6px', borderRadius: '4px', fontWeight: 500 }}>
              Idle
            </span>
          )}
        </div>

      </div>

      {/* Description */}
      <p style={{ margin: 0, fontSize: '12px', color: '#9ca3af', lineHeight: '1.4' }}>
        {agent.description}
      </p>

      {/* Trigger Context/Reason */}
      {agent.trigger_reason && (() => {
        const isLive = agent.is_live_message;
        const hasError = isError;
        let bgColor = 'rgba(255, 255, 255, 0.02)';
        let borderColor = 'rgba(255, 255, 255, 0.04)';
        let textColor = '#9ca3af';
        let dotColor = null;
        let label = 'Last Active';

        if (hasError) {
          bgColor = 'rgba(239, 68, 68, 0.05)';
          borderColor = 'rgba(239, 68, 68, 0.15)';
          textColor = '#f87171';
          label = 'Error';
        } else if (isLive) {
          bgColor = 'rgba(74, 222, 128, 0.06)';
          borderColor = 'rgba(74, 222, 128, 0.25)';
          textColor = '#86efac';
          dotColor = '#4ade80';
          label = 'Active Now';
        }

        return (
          <div style={{
            background: bgColor,
            border: `1px solid ${borderColor}`,
            borderRadius: '6px',
            padding: '6px 10px',
            fontSize: '11px',
            color: textColor,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            transition: 'all 0.3s ease'
          }}>
            {hasError ? (
              <AlertCircle size={12} color="#ef4444" />
            ) : isLive ? (
              <div style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: dotColor,
                boxShadow: `0 0 6px ${dotColor}`,
                animation: 'pulse 1.5s infinite',
                flexShrink: 0
              }} />
            ) : (
              <Clock size={12} color="#6b7280" />
            )}
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
              <span style={{ fontWeight: 700, marginRight: '4px', opacity: 0.7 }}>{label}:</span>
              {hasError ? (agent.last_error || 'Execution failed') : agent.trigger_reason}
            </span>
          </div>
        );
      })()}

      {/* Performance Mini Stats */}
      {agent.invocation_count > 0 && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px dashed rgba(255,255,255,0.03)', paddingTop: '6px', marginTop: '2px', fontSize: '10px', color: '#6b7280' }}>
          <span>Invoked: <strong>{agent.invocation_count}</strong> times</span>
          {agent.last_invoked && (
            <span>Last: {agent.last_invoked.split('T')[1] || agent.last_invoked}</span>
          )}
        </div>
      )}

    </div>
  );
}
