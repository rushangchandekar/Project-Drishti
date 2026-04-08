'use client';
import { motion } from 'framer-motion';
import { Flame, AlertTriangle, Users, Flag } from 'lucide-react';

export default function SpatialMap({ status, activeCamera }) {
  // 3x3 Grid Method for zone-wise detection parsing backend data
  const gridZones = [
    { id: 'top_left', label: 'Top Left' },
    { id: 'top_center', label: 'Top Center' },
    { id: 'top_right', label: 'Top Right' },
    { id: 'mid_left', label: 'Mid Left' },
    { id: 'mid_center', label: 'Mid Center' },
    { id: 'mid_right', label: 'Mid Right' },
    { id: 'bot_left', label: 'Bottom Left' },
    { id: 'bot_center', label: 'Bottom Center' },
    { id: 'bot_right', label: 'Bottom Right' }
  ];

  return (
    <div style={{ flex: 1, position: 'relative', borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: '1px solid var(--border-primary)', background: '#000' }}>
      
      {/* Base Feed - Always use backend processed MJPEG stream */}
      <img src="http://localhost:8000/video-feed" alt="Camera Feed" style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8 }} />

      {/* Grid Overlay for Detection (3x3) */}
      <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gridTemplateRows: 'repeat(3, 1fr)' }}>
        {gridZones.map((zone) => {
          const count = status?.zones?.[zone.id] || 0;
          
          // Determine color based on density in the zone
          let color = 'rgba(255, 255, 255, 0.05)'; // Normal
          let borderColor = 'rgba(255, 255, 255, 0.1)';
          let textColor = 'var(--text-muted)';
          
          if (count > 8) {
            color = 'rgba(239, 68, 68, 0.3)'; // Critical Red
            borderColor = 'rgba(239, 68, 68, 0.6)';
            textColor = 'var(--status-critical)';
          } else if (count > 4) {
            color = 'rgba(245, 158, 11, 0.2)'; // Warning Orange
            borderColor = 'rgba(245, 158, 11, 0.4)';
            textColor = 'var(--status-warning)';
          } else if (count > 0) {
            color = 'rgba(16, 185, 129, 0.1)'; // Active Emerald
            borderColor = 'rgba(16, 185, 129, 0.3)';
            textColor = 'var(--accent-emerald)';
          }

          return (
            <motion.div
              key={zone.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, backgroundColor: color }}
              transition={{ duration: 0.3 }}
              style={{
                border: `1px solid ${borderColor}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
              }}
            >
              {/* Zone Tag */}
              <div style={{ 
                position: 'absolute', 
                top: '8px', left: '8px', 
                background: 'rgba(0,0,0,0.6)', 
                backdropFilter: 'blur(4px)', 
                padding: '4px 8px', 
                borderRadius: '4px', 
                color: 'white', 
                fontSize: '10px', 
                fontWeight: 600,
                border: '1px solid rgba(255,255,255,0.1)'
              }}>
                {zone.label}
              </div>

              {/* Count Indicator */}
              {count > 0 && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: 'rgba(0,0,0,0.8)',
                  padding: '6px 12px',
                  borderRadius: '20px',
                  border: `1px solid ${borderColor}`
                }}>
                  <Users size={14} color={textColor} />
                  <span style={{ color: 'white', fontWeight: 'bold', fontSize: '14px' }}>
                    {count}
                  </span>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

    </div>
  );
}
