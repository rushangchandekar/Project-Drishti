'use client';
import { Users, BarChart2, AlertTriangle, Flame } from 'lucide-react';

export default function AnalyticsCards({ status }) {
  const metrics = {
    crowd_count: status?.person_count || 0,
    density_level: status?.density_level || 'LOW',
    risk_score: status?.risk_score || 0,
    fire_status: status?.fire_detected ? 'ACTIVE' : 'CLEAR'
  };

  const getDensityColor = (level) => {
    switch (level) {
      case 'HIGH':
      case 'CRITICAL':
        return '#ef4444';
      case 'MODERATE': return '#f59e0b';
      case 'LOW': default: return '#4ade80';
    }
  };

  const getRiskColor = (score) => {
    if (score > 60) return '#ef4444';
    if (score > 30) return '#f59e0b';
    return '#4ade80';
  };

  return (
    <div style={{ width: '100%', display: 'flex', gap: '20px', overflowX: 'auto', paddingBottom: '8px' }}>
      
      {/* Crowd Count Card */}
      <Card>
        <CardHeader icon={<Users size={16} color="#8b5cf6" />} title="Crowd Count" />
        <div style={{ fontSize: '32px', fontWeight: 800, color: '#38bdf8', marginTop: '12px', marginBottom: '8px' }}>
          {metrics.crowd_count}
        </div>
        <div style={{ fontSize: '11px', color: '#6b7280', fontWeight: 500 }}>
          People detected
        </div>
      </Card>

      {/* Density Level Card */}
      <Card>
        <CardHeader icon={<BarChart2 size={16} color="#a78bfa" />} title="Density Level" />
        <div style={{ fontSize: '28px', fontWeight: 800, color: getDensityColor(metrics.density_level), marginTop: '12px', marginBottom: '8px', letterSpacing: '1px' }}>
          {metrics.density_level}
        </div>
        <div style={{ fontSize: '11px', color: '#6b7280', fontWeight: 500 }}>
          Current density status
        </div>
      </Card>

      {/* Risk Score Card */}
      <Card>
        <CardHeader icon={<AlertTriangle size={16} color="#f59e0b" />} title="Risk Score" />
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px', marginTop: '12px', marginBottom: '8px' }}>
          <span style={{ fontSize: '32px', fontWeight: 800, color: getRiskColor(metrics.risk_score) }}>
            {metrics.risk_score}
          </span>
          <span style={{ fontSize: '16px', color: '#6b7280', fontWeight: 700 }}>
            /100
          </span>
        </div>
      </Card>

      {/* Fire Status Card */}
      <Card>
        <CardHeader icon={<Flame size={16} color="#f97316" />} title="Fire Status" />
        <div style={{ fontSize: '28px', fontWeight: 800, color: metrics.fire_status === 'CLEAR' ? '#4ade80' : '#ef4444', marginTop: '12px', marginBottom: '8px', letterSpacing: '1px' }}>
          {metrics.fire_status}
        </div>
      </Card>

    </div>
  );
}

function Card({ children }) {
  return (
    <div style={{ 
      flex: 1, 
      minWidth: '220px',
      background: 'var(--bg-secondary)', 
      borderRadius: '12px', 
      padding: '20px', 
      border: '1px solid var(--border-primary)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
    }}>
      {children}
    </div>
  );
}

function CardHeader({ icon, title }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#9ca3af', fontSize: '13px', fontWeight: 500 }}>
      {icon}
      <span>{title}</span>
    </div>
  );
}
