'use client';
import { useState } from 'react';
import { HelpCircle, AlertTriangle, Flame, ShieldAlert } from 'lucide-react';

export default function AlertPanel({ alerts }) {
  const [filter, setFilter] = useState('all');

  const getIcon = (title) => {
    if (title.toLowerCase().includes('fire')) return <Flame size={16} />;
    if (title.toLowerCase().includes('medical') || title.toLowerCase().includes('critical')) return <ShieldAlert size={16} />;
    return <AlertTriangle size={16} />;
  };

  const getColor = (type) => {
    return type === 'critical' ? 'var(--status-critical)' : 'var(--status-warning)';
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'all') return true;
    return alert.type === filter;
  });

  return (
    <div style={{ width: 'var(--alerts-width)', height: '100%', display: 'flex', flexDirection: 'column', gap: '16px', background: 'var(--bg-secondary)', borderLeft: '1px solid var(--border-primary)', padding: '20px' }}>
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 style={{ fontSize: '15px', fontWeight: 600 }}>Real-Time Alerts</h2>
        <button style={{ background: 'rgba(167, 139, 250, 0.1)', color: 'var(--accent-purple)', border: '1px solid rgba(167, 139, 250, 0.2)', display: 'flex', alignItems: 'center', gap: '4px', padding: '4px 10px', borderRadius: '100px', fontSize: '11px', cursor: 'pointer' }}>
          <HelpCircle size={12} />
          Help
        </button>
      </div>

      <div style={{ display: 'flex', gap: '8px' }}>
        <button 
          onClick={() => setFilter('all')}
          style={{ cursor: 'pointer', padding: '4px 12px', borderRadius: '100px', fontSize: '12px', background: filter === 'all' ? 'var(--accent-blue)' : 'transparent', color: filter === 'all' ? 'white' : 'var(--text-secondary)', border: filter === 'all' ? 'none' : '1px solid var(--border-primary)' }}
        >All</button>
        <button 
          onClick={() => setFilter('critical')}
          style={{ cursor: 'pointer', padding: '4px 12px', borderRadius: '100px', fontSize: '12px', background: filter === 'critical' ? 'var(--accent-blue)' : 'transparent', color: filter === 'critical' ? 'white' : 'var(--text-secondary)', border: filter === 'critical' ? 'none' : '1px solid var(--border-primary)' }}
        >Critical</button>
        <button 
          onClick={() => setFilter('warning')}
          style={{ cursor: 'pointer', padding: '4px 12px', borderRadius: '100px', fontSize: '12px', background: filter === 'warning' ? 'var(--accent-blue)' : 'transparent', color: filter === 'warning' ? 'white' : 'var(--text-secondary)', border: filter === 'warning' ? 'none' : '1px solid var(--border-primary)' }}
        >Warning</button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px', paddingRight: '4px' }}>
        {filteredAlerts.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px', marginTop: '24px' }}>No {filter !== 'all' ? filter : ''} alerts at this time.</div>
        )}
        {filteredAlerts.map(alert => (
          <div key={alert.id} style={{ 
            padding: '16px', 
            background: 'rgba(23, 27, 43, 0.8)', 
            borderRadius: '6px', 
            borderLeft: `4px solid ${alert.type === 'critical' ? 'rgba(239,68,68,1)' : '#f59e0b'}`,
            display: 'flex', 
            flexDirection: 'column', 
            gap: '12px' 
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ color: getColor(alert.type), fontWeight: 600, fontSize: '14px' }}>
                {alert.title}
              </div>
              <span style={{ fontSize: '12px', color: 'rgba(156, 163, 175, 1)' }}>{alert.time}</span>
            </div>
            <p style={{ fontSize: '13px', color: 'rgba(229, 231, 235, 1)', lineHeight: 1.5, margin: 0 }}>
              {alert.message}
            </p>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '4px' }}>
              <span style={{ 
                padding: '4px 10px', 
                borderRadius: '100px', 
                background: alert.type === 'critical' ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.15)', 
                color: getColor(alert.type), 
                fontSize: '11px', 
                fontWeight: 600
              }}>
                {alert.type.charAt(0).toUpperCase() + alert.type.slice(1)}
              </span>
              <span style={{ 
                padding: '4px 10px', 
                borderRadius: '6px', 
                background: 'rgba(37, 99, 235, 0.15)', 
                color: '#60a5fa', 
                fontSize: '11px', 
                fontWeight: 600 
              }}>
                {alert.agent}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
