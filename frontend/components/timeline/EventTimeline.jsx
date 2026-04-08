'use client';
import { Clock, Filter } from 'lucide-react';

export default function EventTimeline({ events }) {
  return (
    <div style={{ width: '100%', height: 'var(--timeline-height)', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border-primary)', padding: '16px 24px', display: 'flex', flexDirection: 'column' }}>
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, fontSize: '14px' }}>
          <Clock size={16} />
          Event Timeline
        </div>
        <button style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'transparent', border: '1px solid var(--border-primary)', color: 'var(--text-secondary)', padding: '4px 12px', borderRadius: '6px', fontSize: '12px' }}>
          <Filter size={12} />
          Last 30 minutes
        </button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', position: 'relative' }}>
        {/* Timeline connector line */}
        <div style={{ position: 'absolute', left: '3px', top: '10px', bottom: '10px', width: '2px', background: 'var(--border-primary)', zIndex: 0 }} />

        {events.map((event, idx) => (
          <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', padding: '6px 0', position: 'relative', zIndex: 1 }}>
            {/* Dot */}
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: event.statusColor, border: '2px solid var(--bg-secondary)', marginTop: '4px', flexShrink: 0, boxShadow: `0 0 8px ${event.statusColor}` }} />
            
            {/* Content */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '13px' }}>
              <span style={{ color: event.statusColor, fontWeight: 600 }}>{event.type}</span>
              <span style={{ color: 'var(--text-muted)' }}>{event.time}</span>
              <span style={{ color: 'var(--text-primary)' }}>- {event.message}</span>
              {event.status && (
                <span style={{ background: `color-mix(in srgb, ${event.statusColor} 15%, transparent)`, color: event.statusColor, padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600, border: `1px solid color-mix(in srgb, ${event.statusColor} 30%, transparent)` }}>
                  {event.status}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
