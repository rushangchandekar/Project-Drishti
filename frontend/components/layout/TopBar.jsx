'use client';
import { useState } from 'react';
import { MapPin, Clock, AlertTriangle, Radio, RefreshCw, PenLine } from 'lucide-react';

export default function TopBar({ liveTime, status }) {
  const [localVenue, setLocalVenue] = useState(null);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const currentVenue = localVenue || status?.venue_name || 'Downtown Stadium, NYC';

  const handleEditVenue = () => {
    const newVenue = prompt("Enter venue name:", currentVenue);
    if (newVenue && newVenue.trim() !== "") {
      setLocalVenue(newVenue.trim());
    }
  };

  return (
    <header style={{ height: 'var(--topbar-height)', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px', backgroundColor: 'var(--bg-primary)', zIndex: 10, borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      
      {/* Left: Titles */}
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <h1 style={{ fontSize: '20px', fontWeight: 800, color: '#38bdf8', letterSpacing: '4px', margin: 0, lineHeight: 1.2 }}>DRISHTI</h1>
        <span style={{ fontSize: '11px', color: '#9ca3af', fontWeight: 600, letterSpacing: '0.5px' }}>Crowd Intelligence System</span>
      </div>

      {/* Right: Status Pills & Action Buttons */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        
        {/* Location Pill */}
        <div 
          onClick={handleEditVenue}
          style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(255, 255, 255, 0.05)', padding: '6px 14px', borderRadius: '8px', fontSize: '13px', color: '#9ca3af', fontWeight: 500, cursor: 'pointer', outline: 'none' }}>
          <MapPin size={14} />
          {currentVenue}
          <PenLine size={12} style={{ marginLeft: '4px', opacity: 0.7 }} />
        </div>

        {/* Live Timer Pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(22, 163, 74, 0.15)', padding: '6px 16px', borderRadius: '8px', fontSize: '12px', color: '#4ade80', fontWeight: 700, letterSpacing: '1px', border: '1px solid rgba(22, 163, 74, 0.3)' }}>
          <div style={{ width: '8px', height: '8px', backgroundColor: '#4ade80', borderRadius: '50%' }} className="animate-pulse" />
          LIVE
          <Clock size={14} style={{ marginLeft: '6px' }} />
          {formatTime(liveTime)}
        </div>

        {/* System Online Pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(255, 255, 255, 0.05)', padding: '6px 14px', borderRadius: '8px', fontSize: '12px', color: '#4ade80', fontWeight: 600 }}>
          <Radio size={14} />
          System Online
        </div>

        <div style={{ width: '1px', height: '24px', backgroundColor: 'rgba(255,255,255,0.1)', margin: '0 4px' }} />

        {/* Refresh Button */}
        <button 
          onClick={() => window.location.reload()}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '36px', height: '36px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)', color: '#9ca3af', cursor: 'pointer', transition: 'all 0.2s ease' }} 
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)'} 
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)'}
        >
          <RefreshCw size={16} />
        </button>

        {/* Manual Override Subsystem */}
        <button onClick={() => alert("Manual Override Triggered!")} style={{ display: 'flex', alignItems: 'center', gap: '8px', borderRadius: '8px', padding: '0 16px', height: '36px', fontSize: '13px', fontWeight: 700, backgroundColor: '#ef4444', border: 'none', color: '#ffffff', cursor: 'pointer', transition: 'transform 0.1s ease', boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)' }} onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'} onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}>
          <AlertTriangle size={16} color="#ffffff" />
          Manual Override
        </button>
      </div>
    </header>
  );
}
