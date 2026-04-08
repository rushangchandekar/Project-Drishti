'use client';
import { Home, LineChart, Brain, Bell, Settings, LogOut } from 'lucide-react';

export default function Sidebar({ activeTab, onTabChange, unreadCount }) {
  return (
    <aside style={{ width: 'var(--sidebar-width)', height: '100%', borderRight: '1px solid var(--border-primary)', backgroundColor: 'var(--bg-secondary)', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '24px 0', zIndex: 10 }}>
      {/* Logo Removed */}

      {/* Nav */}
      <nav style={{ display: 'flex', flexDirection: 'column', gap: '24px', flex: 1, width: '100%', alignItems: 'center' }}>
        <NavItem icon={<Home size={20} />} active={activeTab === 'home'} onClick={() => onTabChange('home')} />
        <NavItem icon={<LineChart size={20} />} active={activeTab === 'analytics'} onClick={() => onTabChange('analytics')} />
        <NavItem icon={<Brain size={20} />} active={activeTab === 'ai'} onClick={() => onTabChange('ai')} />
        <NavItem icon={<Bell size={20} />} active={activeTab === 'alerts'} onClick={() => onTabChange('alerts')} badge={unreadCount > 0 ? unreadCount : null} />
        <NavItem icon={<Settings size={20} />} active={activeTab === 'settings'} onClick={() => onTabChange('settings')} />
      </nav>

      {/* Footer */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', alignItems: 'center' }}>
        <NavItem icon={<LogOut size={20} />} onClick={() => alert("Logged out")} />
      </div>
    </aside>
  );
}

function NavItem({ icon, active, badge, onClick }) {
  return (
    <div onClick={onClick} style={{ position: 'relative', width: '40px', height: '40px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: active ? 'white' : 'var(--text-secondary)', background: active ? 'rgba(255,255,255,0.1)' : 'transparent', cursor: 'pointer', transition: 'all 0.2s', border: active ? '1px solid rgba(255,255,255,0.05)' : '1px solid transparent' }}>
      {active && <div style={{ position: 'absolute', left: '-12px', width: '4px', height: '24px', background: 'var(--accent-blue)', borderRadius: '0 4px 4px 0' }} />}
      {icon}
      {badge && (
        <span style={{ position: 'absolute', top: '-4px', right: '-4px', background: 'var(--status-critical)', color: 'white', fontSize: '10px', fontWeight: 700, width: '16px', height: '16px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {badge}
        </span>
      )}
    </div>
  );
}
