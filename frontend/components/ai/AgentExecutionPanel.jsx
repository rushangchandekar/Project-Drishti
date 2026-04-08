'use client';
import { BrainCircuit, Activity, Network, CheckCircle2, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function AgentExecutionPanel({ agentFeed }) {
  // We simulate continuous execution flow if agentFeed is empty, otherwise we map the actual feed
  const liveFeed = (agentFeed && agentFeed.length > 0) ? agentFeed : [
    { id: 1, agent: 'CrowdAnalyzer', status: 'Running', detail: 'Parsing density zones', time: 'Just now', icon: <Activity size={18} color="#38bdf8" /> },
    { id: 2, agent: 'AnomalyDetector', status: 'Standby', detail: 'Awaiting threshold triggers', time: '1m ago', icon: <Network size={18} color="#a78bfa" /> },
    { id: 3, agent: 'n8n Workflow', status: 'Completed', detail: 'Webhook payload sent to external system', time: '5m ago', icon: <CheckCircle2 size={18} color="#4ade80" /> }
  ];

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)', borderRadius: '16px', border: '1px solid var(--border-primary)', overflow: 'hidden' }}>
      
      {/* Header */}
      <div style={{ padding: '24px', borderBottom: '1px solid var(--border-primary)', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <BrainCircuit size={28} color="#38bdf8" />
        <div>
          <h2 style={{ margin: 0, fontSize: '20px', fontWeight: 700, color: 'white' }}>Live Agent Execution</h2>
          <span style={{ fontSize: '13px', color: '#9ca3af' }}>n8n Workflows & AI Intelligence Feed</span>
        </div>
      </div>

      {/* Main Execution Flow Area */}
      <div style={{ flex: 1, padding: '24px', overflowY: 'auto', display: 'flex', gap: '24px' }}>
        
        {/* Left Status Nodes (Representing Core Agents) */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontSize: '14px', color: '#9ca3af', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>Core Agents (Continuous)</h3>
          
          <AgentNode name="Computer Vision Agent" type="YOLOv8 Processing" status="Active" color="#38bdf8" />
          <AgentNode name="Anomaly Detection Agent" type="Behavior Analysis" status="Active" color="#a78bfa" />
          <AgentNode name="Fire Safety Agent" type="IR/Visual Scan" status="Standby" color="#f59e0b" />
          
        </div>

        {/* Vertical Divider */}
        <div style={{ width: '1px', backgroundColor: 'var(--border-primary)' }} />

        {/* Right Event Log (n8n execution & real actions) */}
        <div style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ fontSize: '14px', color: '#9ca3af', marginBottom: '24px', textTransform: 'uppercase', letterSpacing: '1px' }}>Action & Workflow Trace</h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <AnimatePresence>
              {liveFeed.map((event, idx) => (
                <motion.div 
                  key={event.id || idx}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  style={{ display: 'flex', gap: '16px', background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}
                >
                  <div style={{ marginTop: '2px' }}>
                    {event.icon || <Activity size={18} color="#4ade80" />}
                  </div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <span style={{ fontWeight: 600, color: 'white', fontSize: '15px' }}>{event.agent || 'System Agent'}</span>
                      <span style={{ fontSize: '12px', padding: '2px 8px', borderRadius: '4px', background: event.status === 'Completed' ? 'rgba(74, 222, 128, 0.1)' : 'rgba(56, 189, 248, 0.1)', color: event.status === 'Completed' ? '#4ade80' : '#38bdf8' }}>
                        {event.status || 'Triggered'}
                      </span>
                    </div>
                    <p style={{ margin: 0, fontSize: '13px', color: '#9ca3af', lineHeight: 1.5 }}>{event.action || event.detail}</p>
                    <span style={{ fontSize: '11px', color: '#6b7280', marginTop: '6px', display: 'block' }}>{event.timestamp || event.time}</span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

      </div>
    </div>
  );
}

function AgentNode({ name, type, status, color }) {
  return (
    <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ width: '40px', height: '40px', borderRadius: '8px', background: `${color}20`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <BrainCircuit size={20} color={color} />
        </div>
        <div>
          <h4 style={{ margin: 0, color: 'white', fontSize: '14px' }}>{name}</h4>
          <span style={{ color: '#6b7280', fontSize: '12px' }}>{type}</span>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: status === 'Active' ? '#4ade80' : '#f59e0b' }} className={status === 'Active' ? 'animate-pulse' : ''} />
        <span style={{ fontSize: '12px', color: status === 'Active' ? '#4ade80' : '#f59e0b', fontWeight: 600 }}>{status}</span>
      </div>
    </div>
  );
}
