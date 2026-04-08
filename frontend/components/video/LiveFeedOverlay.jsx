'use client';
import { motion } from 'framer-motion';

export default function LiveFeedOverlay({ activeCamera }) {
  return (
    <motion.div 
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ delay: 0.5 }}
      style={{ 
        position: 'absolute', 
        bottom: '24px', 
        left: '24px', 
        width: '280px', 
        height: '160px', 
        borderRadius: '12px', 
        overflow: 'hidden', 
        border: '1px solid var(--border-primary)', 
        boxShadow: 'var(--shadow-lg)',
        background: '#000'
      }}
    >
      <img 
        src="http://localhost:8000/video-feed" 
        alt="Live Drone Feed" 
        style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
      />
      <div style={{ position: 'absolute', top: '8px', left: '8px', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', padding: '2px 8px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: 'white', fontWeight: 600 }}>
        <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--status-critical)' }} className="animate-pulse" />
        LIVE: {activeCamera?.name || "Drone #3"}
      </div>
      <div style={{ position: 'absolute', bottom: '8px', left: '8px', right: '8px', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', padding: '6px 8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)', fontSize: '11px', color: 'white', display: 'flex', alignItems: 'center', gap: '6px' }}>
         <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{activeCamera?.agent || "VisionAgent"}:</span> Observing Area
      </div>
    </motion.div>
  );
}
