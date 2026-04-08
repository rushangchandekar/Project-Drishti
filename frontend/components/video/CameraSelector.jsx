'use client';
import { Camera, Radio } from 'lucide-react';

export default function CameraSelector({ cameras, activeCamera, onSelect }) {
  return (
    <div style={{ width: '100%', height: 'var(--timeline-height)', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border-primary)', padding: '16px 24px', display: 'flex', flexDirection: 'column' }}>
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)', fontWeight: 600, fontSize: '14px' }}>
          <Radio size={16} color="var(--status-critical)" className="animate-pulse" />
          Active Feeds ({cameras.length})
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', gap: '16px', overflowX: 'auto', paddingBottom: '8px', alignItems: 'center' }}>
        {cameras.map(cam => {
          const isActive = activeCamera?.id === cam.id;
          return (
            <div 
              key={cam.id} 
              onClick={() => onSelect(cam)}
              style={{
                position: 'relative',
                minWidth: '200px',
                height: '100%',
                borderRadius: '8px',
                overflow: 'hidden',
                cursor: 'pointer',
                border: isActive ? '2px solid var(--accent-blue)' : '1px solid var(--border-primary)',
                transition: 'all 0.2s ease',
                boxShadow: isActive ? '0 0 12px rgba(59, 130, 246, 0.4)' : 'none',
                opacity: isActive ? 1 : 0.6
              }}
            >
               {cam.type === 'file' ? (
                 <video src={cam.src} style={{ width: '100%', height: '100%', objectFit: 'cover' }} muted loop playsInline autoPlay />
               ) : (
                 <img src={cam.src} alt={cam.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { e.target.src = 'https://images.unsplash.com/photo-1508614589041-895b88991e3e?auto=format&fit=crop&q=80' }} />
               )}
               <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(180deg, rgba(0,0,0,0) 50%, rgba(0,0,0,0.8) 100%)' }} />
               
               <div style={{ position: 'absolute', bottom: '8px', left: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <div style={{ color: 'white', fontSize: '12px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Camera size={12} /> {cam.name}
                  </div>
                  <div style={{ color: 'var(--accent-emerald)', fontSize: '10px', fontWeight: 600 }}>
                    {cam.agent}
                  </div>
               </div>
               
               {isActive && (
                 <div style={{ position: 'absolute', top: '8px', right: '8px', width: '8px', height: '8px', borderRadius: '50%', background: 'var(--status-critical)' }} className="animate-pulse" />
               )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
