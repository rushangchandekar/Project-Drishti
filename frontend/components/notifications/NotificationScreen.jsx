'use client';
import { BellRing, ShieldAlert, CheckCircle2, Siren, Zap, Flame, Users } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function NotificationScreen({ alerts, autonomousActions }) {
  // If no alerts yet, we can show a placeholder or let it be empty
  const hasAlerts = alerts && alerts.length > 0;
  
  return (
    <div style={{ flex: 1, display: 'flex', gap: '24px', height: '100%' }}>
      
      {/* Left Area: General Alerts */}
      <div style={{ flex: 2, display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)', borderRadius: '16px', border: '1px solid var(--border-primary)', overflow: 'hidden' }}>
        <div style={{ padding: '24px', borderBottom: '1px solid var(--border-primary)', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <BellRing size={28} color="#f59e0b" />
          <div>
            <h2 style={{ margin: 0, fontSize: '20px', fontWeight: 700, color: 'white' }}>System Notifications</h2>
            <span style={{ fontSize: '13px', color: '#9ca3af' }}>Chronological Event Log</span>
          </div>
        </div>

        <div style={{ flex: 1, padding: '24px', overflowY: 'auto' }}>
          {!hasAlerts ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#6b7280' }}>
              No critical notifications at this time.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <AnimatePresence>
                {alerts.map((alert, idx) => (
                  <motion.div 
                    key={alert.id || idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ 
                      background: alert.type === 'critical' ? 'rgba(239, 68, 68, 0.05)' : 'rgba(255, 255, 255, 0.02)', 
                      border: `1px solid ${alert.type === 'critical' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(255,255,255,0.05)'}`, 
                      borderRadius: '12px', 
                      padding: '16px',
                      display: 'flex',
                      gap: '16px'
                    }}
                  >
                    <div style={{ marginTop: '2px' }}>
                      {alert.type === 'critical' && alert.title.includes('Fire') ? <Flame size={20} color="#ef4444" /> : 
                       alert.type === 'critical' ? <Users size={20} color="#ef4444" /> : 
                       <ShieldAlert size={20} color="#f59e0b" />}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                        <h4 style={{ margin: 0, color: alert.type === 'critical' ? '#ef4444' : '#f59e0b', fontSize: '15px' }}>{alert.title}</h4>
                        <span style={{ fontSize: '12px', color: '#6b7280' }}>{alert.time}</span>
                      </div>
                      <p style={{ margin: 0, fontSize: '13px', color: '#d1d5db', lineHeight: 1.5 }}>{alert.message}</p>
                      <div style={{ display: 'inline-block', marginTop: '12px', fontSize: '11px', padding: '4px 8px', borderRadius: '4px', background: 'rgba(255,255,255,0.05)', color: '#9ca3af' }}>
                        Detected by: {alert.agent}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </div>
      </div>

      {/* Right Area: Autonomous Actions */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)', borderRadius: '16px', border: '1px solid rgba(56, 189, 248, 0.2)', overflow: 'hidden', boxShadow: '0 0 20px rgba(56, 189, 248, 0.05)' }}>
        <div style={{ padding: '24px', background: 'rgba(56, 189, 248, 0.05)', borderBottom: '1px solid rgba(56, 189, 248, 0.1)', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Zap size={24} color="#38bdf8" />
          <div>
            <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'white' }}>Autonomous Actions</h2>
            <span style={{ fontSize: '12px', color: '#38bdf8' }}>Physical & Network Responses</span>
          </div>
        </div>
        
        <div style={{ flex: 1, padding: '24px', overflowY: 'auto' }}>
          {autonomousActions && autonomousActions.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <AnimatePresence>
                {autonomousActions.map((action, idx) => (
                  <motion.div 
                    key={idx}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    style={{ background: 'rgba(56, 189, 248, 0.05)', borderRadius: '12px', padding: '16px', border: '1px solid rgba(56, 189, 248, 0.1)' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                      <Siren size={18} color="#38bdf8" style={{ marginTop: '2px' }} />
                      <div>
                        <span style={{ fontSize: '14px', color: 'white', fontWeight: 500, display: 'block', marginBottom: '4px' }}>{action}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <CheckCircle2 size={12} color="#4ade80" />
                          <span style={{ fontSize: '11px', color: '#4ade80' }}>Executed</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#6b7280', textAlign: 'center', gap: '12px' }}>
              <ShieldAlert size={32} style={{ opacity: 0.2 }} />
              <span style={{ fontSize: '13px' }}>No autonomous physical safeguards<br/>have been deployed yet.</span>
            </div>
          )}
        </div>
      </div>
      
    </div>
  );
}
