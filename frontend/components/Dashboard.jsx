'use client';
import { useState, useEffect } from 'react';
import Sidebar from './layout/Sidebar';
import TopBar from './layout/TopBar';
import SpatialMap from './map/SpatialMap';
import AlertPanel from './alerts/AlertPanel';
import CameraSelector from './video/CameraSelector';
import LiveFeedOverlay from './video/LiveFeedOverlay';
import GeminiChat from './chat/GeminiChat';
import AnalyticsCards from './analytics/AnalyticsCards';
import AgentExecutionPanel from './ai/AgentExecutionPanel';
import NotificationScreen from './notifications/NotificationScreen';
import { useSystemStatus } from '../hooks/useSystemStatus';

export default function Dashboard() {
  const { status, alerts, agentFeed, autonomousActions, agentStatuses } = useSystemStatus(true);
  const [liveTime, setLiveTime] = useState(0); 
  const [cameras, setCameras] = useState([{ id: 'cam0', name: 'Main Gate (Webcam)', type: 'webcam', path: '0', agent: 'VisionAgent', src: 'http://localhost:8000/video-feed' }]);
  const [activeCamera, setActiveCamera] = useState(null);
  const [activeTab, setActiveTab] = useState('home');
  const [lastSeenAlertId, setLastSeenAlertId] = useState(null);

  // Fetch available videos from backend
  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const response = await fetch('http://localhost:8000/list-videos');
        const data = await response.json();
        const videoCameras = (data.videos || []).map((vid, i) => ({
          id: `vid-${i}`,
          name: vid.name.replace(/\.[^/.]+$/, ""), // remove extension for name
          src: `http://localhost:8000/data/${vid.name}`,
          type: 'file',
          path: vid.path,
          agent: vid.name.toLowerCase().includes('crowd') ? 'CrowdAgent' : 'VisionAgent'
        }));
        
        const allCameras = [
          { id: 'cam0', name: 'Main Gate (Webcam)', type: 'webcam', path: '0', agent: 'VisionAgent', src: 'http://localhost:8000/video-feed' },
          ...videoCameras
        ];
        
        setCameras(allCameras);
        setActiveCamera(allCameras[0]); // Default to webcam
      } catch (err) {
        console.error("Failed to fetch videos from server", err);
        setActiveCamera({ id: 'cam0', name: 'Main Gate', type: 'webcam', path: '0', agent: 'VisionAgent', src: 'http://localhost:8000/video-feed' });
      }
    };
    fetchVideos();
  }, []);

  const handleCameraSelect = async (cam) => {
    try {
      await fetch('http://localhost:8000/switch-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: cam.type,
          path: cam.path
        })
      });
      setActiveCamera(cam);
    } catch (e) {
      console.error("Failed to switch camera", e);
    }
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setLiveTime(prev => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Update last seen alert when viewing alerts tab
  useEffect(() => {
    if (activeTab === 'alerts' && alerts && alerts.length > 0) {
      setLastSeenAlertId(alerts[0].id);
    }
  }, [alerts, activeTab]);

  const unreadCount = (() => {
    if (activeTab === 'alerts') return 0;
    if (!alerts || alerts.length === 0) return 0;
    if (!lastSeenAlertId) return alerts.length;
    
    const index = alerts.findIndex(a => a.id === lastSeenAlertId);
    if (index === -1) return alerts.length;
    return index;
  })();

  return (
    <div className="dashboard-shell" style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', overflow: 'hidden', backgroundColor: 'var(--bg-primary)' }}>
      <TopBar liveTime={liveTime} status={status} />
      
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <Sidebar activeTab={activeTab} onTabChange={setActiveTab} unreadCount={unreadCount} />
        
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          
          {activeTab === 'ai' ? (
            <div style={{ flex: 1, display: 'flex', padding: '20px', overflow: 'hidden' }}>
              <AgentExecutionPanel agentFeed={agentFeed} agentStatuses={agentStatuses} liveStatus={status} />
            </div>
          ) : activeTab === 'alerts' ? (
            <div style={{ flex: 1, display: 'flex', padding: '20px', overflow: 'hidden' }}>
              <NotificationScreen alerts={alerts} autonomousActions={autonomousActions} />
            </div>
          ) : (
            <>
              {/* Main View Area (Map + Alerts) */}
              <div style={{ flex: 1, display: 'flex', gap: '20px', padding: '20px', overflow: 'hidden' }}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
                  <SpatialMap status={status} activeCamera={activeCamera} />
                  <LiveFeedOverlay activeCamera={activeCamera} />
                </div>
                {activeTab === 'home' && <AlertPanel alerts={alerts} />}
              </div>

              {/* Bottom Panel Conditional Rendering */}
              <div style={{ padding: '0 20px 20px 20px' }}>
                {activeTab === 'analytics' ? (
                  <AnalyticsCards status={status} />
                ) : (
                  <CameraSelector cameras={cameras} activeCamera={activeCamera} onSelect={handleCameraSelect} />
                )}
              </div>
            </>
          )}
          
        </div>
      </div>

      <GeminiChat />

    </div>
  );
}
