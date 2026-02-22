import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Home, BarChart3, Bot, Bell, Settings, AlertTriangle,
  Flame, Users, Shield, Activity, Clock, MapPin,
  Send, X, Maximize2,
  Radio, Eye, Zap, AlertCircle, CheckCircle,
  RefreshCw, MessageCircle, Camera,
  Minimize2, HelpCircle, Webcam, Monitor,
  RotateCcw, Sparkles, Plus, Edit2, Check
} from 'lucide-react';

function App() {
  // State management
  const [status, setStatus] = useState({
    person_count: 0,
    density_level: 'UNKNOWN',
    density_value: 0,
    risk_score: 0,
    fire_detected: false,
    fire_confidence: 0,
    anomaly_detected: false,
    anomaly_type: null,
    trend: 'STABLE',
    zones: {},
    webhooks_sent: 0,
    recommendation: '',
    strategic_guidance: '',
    connected: false,
    detection_time_ms: 0,
    decision_time_ms: 0,
    total_loop_time_ms: 0
  });

  const [alerts, setAlerts] = useState([]);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatExpanded, setChatExpanded] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      type: 'bot',
      text: 'Hello! I\'m your Drishti AI Assistant. I can help you monitor crowd safety, analyze patterns, and provide real-time recommendations.',
      agents: ['SystemAgent']
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  // const [soundEnabled, setSoundEnabled] = useState(true); // Sound removed
  const [liveTime, setLiveTime] = useState(0);
  const [alertFilter, setAlertFilter] = useState('all');
  const [isHeatmapMain, setIsHeatmapMain] = useState(true);
  const [selectedCamera, setSelectedCamera] = useState('webcam');
  const [videoError, setVideoError] = useState(false);
  const [videoKey, setVideoKey] = useState(Date.now());

  // Fetch status from backend
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/status');
        const data = await response.json();

        // Generate alerts based on status changes
        if (data.fire_detected && !status.fire_detected) {
          addAlert('critical', 'Fire Detected!', 'Immediate evacuation required', 'FireAgent');
        }
        if (data.density_level === 'CRITICAL' && status.density_level !== 'CRITICAL') {
          addAlert('critical', 'Critical Density', 'Crowd density has reached critical levels', 'CrowdAgent');
        }
        if (data.anomaly_detected && !status.anomaly_detected) {
          addAlert('warning', 'Anomaly Detected', data.anomaly_type || 'Unusual pattern detected', 'AnomalyAgent');
        }
        if (data.risk_score > 70 && status.risk_score <= 70) {
          addAlert('warning', 'High Risk Alert', `Risk score elevated to ${data.risk_score}`, 'ForecastAgent');
        }

        setStatus({ ...data, connected: true });
      } catch (error) {
        console.error('Status fetch error:', error);
        setStatus(prev => ({ ...prev, connected: false }));
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [status.fire_detected, status.density_level, status.anomaly_detected, status.risk_score]);

  // Live timer
  useEffect(() => {
    const timer = setInterval(() => {
      setLiveTime(prev => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const addAlert = (type, title, message, agent) => {
    const newAlert = {
      id: Date.now(),
      type,
      title,
      message,
      agent,
      time: new Date().toLocaleTimeString()
    };
    setAlerts(prev => [newAlert, ...prev].slice(0, 20));
  };

  const handleChatSend = async () => {
    if (!chatInput.trim() || chatLoading) return;

    const userMessage = { type: 'user', text: chatInput, initials: 'U' };
    setChatMessages(prev => [...prev, userMessage]);
    const question = chatInput;
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });

      if (!response.ok) {
        throw new Error('Query failed');
      }

      const data = await response.json();

      const botMessage = {
        type: 'bot',
        text: data.answer || 'No response received',
        agents: detectAgents(question)
      };
      setChatMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        type: 'bot',
        text: 'I apologize, but I encountered an error. Please ensure the backend is running and Gemini API is configured.',
        agents: ['SystemAgent'],
        isError: true
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setChatLoading(false);
    }
  };

  const detectAgents = (question) => {
    const q = question.toLowerCase();
    const agents = [];
    if (q.includes('crowd') || q.includes('people') || q.includes('count')) agents.push('CrowdAgent');
    if (q.includes('fire') || q.includes('smoke')) agents.push('FireAgent');
    if (q.includes('risk') || q.includes('danger')) agents.push('ForecastAgent');
    if (q.includes('camera') || q.includes('video')) agents.push('VisionAgent');
    if (q.includes('anomal') || q.includes('unusual')) agents.push('AnomalyAgent');
    if (agents.length === 0) agents.push('LLMAgent');
    return agents;
  };

  const handleCameraSwitch = async (cameraId) => {
    setSelectedCamera(cameraId);
    setVideoKey(Date.now()); // Force video refresh

    try {
      await fetch('http://localhost:8000/switch-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: cameraId === 'webcam' ? 'webcam' : 'file',
          path: cameraId
        })
      });
    } catch (error) {
      console.error('Failed to switch camera:', error);
    }
  };

  const refreshVideo = () => {
    setVideoKey(Date.now());
    setVideoError(false);
  };

  const filteredAlerts = alerts.filter(alert => {
    if (alertFilter === 'all') return true;
    return alert.type === alertFilter;
  });

  const getDensityColor = (level) => {
    const colors = {
      'CRITICAL': '#f85149',
      'VERY_HIGH': '#db6d28',
      'HIGH': '#d29922',
      'MODERATE': '#d29922',
      'LOW': '#3fb950',
      'VERY_LOW': '#3fb950',
      'EMPTY': '#8b949e'
    };
    return colors[level] || '#8b949e';
  };

  const swapVideos = () => {
    setIsHeatmapMain(!isHeatmapMain);
  };

  return (
    <div className="dashboard">
      {/* Sidebar removed */}

      <div className="main-content">
        <TopBar
          status={status}
          liveTime={liveTime}
          formatTime={formatTime}
        />

        <div className="workspace">
          <div className="map-container">
            <SpatialMap
              status={status}
              getDensityColor={getDensityColor}
              isHeatmapMain={isHeatmapMain}
              videoKey={videoKey}
              videoError={videoError}
              setVideoError={setVideoError}
              onRefresh={refreshVideo}
            />
          </div>

          <AlertPanel
            alerts={filteredAlerts}
            alertFilter={alertFilter}
            setAlertFilter={setAlertFilter}
          />
        </div>

        <CameraSwitcher
          selectedCamera={selectedCamera}
          onCameraSelect={handleCameraSwitch}
        />
      </div>

      <VideoPreview
        isHeatmapMain={isHeatmapMain}
        onSwap={swapVideos}
        videoKey={videoKey}
      />

      <GlassmorphismChat
        isOpen={chatOpen}
        setIsOpen={setChatOpen}
        isExpanded={chatExpanded}
        setIsExpanded={setChatExpanded}
        messages={chatMessages}
        input={chatInput}
        setInput={setChatInput}
        onSend={handleChatSend}
        isLoading={chatLoading}
        status={status}
      />

      {!chatOpen && (
        <motion.button
          className="chat-toggle-btn"
          onClick={() => setChatOpen(true)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Sparkles size={20} />
          <span>AI Assistant</span>
        </motion.button>
      )}

      <style>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        .dashboard {
          display: flex;
          height: 100vh;
          width: 100vw;
          background: var(--bg-primary);
          overflow-y: auto;
          overflow-x: hidden;
        }

        .main-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          margin-left: 0;
          height: auto;
          min-height: 100vh;
        }

        .workspace {
          flex: 1;
          display: flex;
          padding: 20px;
          gap: 20px;
          overflow: hidden;
        }

        .map-container {
          flex: 1;
          position: relative;
          border-radius: 16px;
          overflow: hidden;
          background: var(--bg-secondary);
          border: 1px solid var(--border-primary);
        }

        .chat-toggle-btn {
          position: fixed;
          bottom: 120px;
          right: 24px;
          padding: 14px 24px;
          border-radius: 50px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border: none;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 10px;
          font-weight: 600;
          font-size: 14px;
          box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
          z-index: 100;
          transition: all 0.2s ease;
        }

        .chat-toggle-btn:hover {
          box-shadow: 0 12px 32px rgba(102, 126, 234, 0.6);
          transform: translateY(-2px);
        }
      `}</style>
    </div>
  );
}



// TopBar Component
function TopBar({ status, liveTime, formatTime }) {
  const [venueName, setVenueName] = useState("Main Venue - Section Alpha");
  const [isEditing, setIsEditing] = useState(false);
  const [tempName, setTempName] = useState(venueName);

  const handleEditClick = () => {
    setTempName(venueName);
    setIsEditing(true);
  };

  const handleSave = () => {
    setVenueName(tempName);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setIsEditing(false);
  };

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1 className="project-title">
          <span className="gradient-text">DRISHTI</span>
          <span className="subtitle">Crowd Intelligence System</span>
        </h1>
      </div>

      <div className="topbar-center">
        <div className="location-badge">
          <MapPin size={16} />
          {isEditing ? (
            <div className="edit-venue-container">
              <input
                type="text"
                value={tempName}
                onChange={(e) => setTempName(e.target.value)}
                className="venue-input"
                autoFocus
              />
              <button onClick={handleSave} className="edit-btn save"><Check size={14} /></button>
              <button onClick={handleCancel} className="edit-btn cancel"><X size={14} /></button>
            </div>
          ) : (
            <div className="venue-display" onClick={handleEditClick} title="Click to edit">
              <span>{venueName}</span>
              <Edit2 size={12} className="edit-icon" />
            </div>
          )}
        </div>

        <div className="live-badge">
          <div className="live-dot" />
          <span>LIVE</span>
          <Clock size={14} />
          <span className="live-time">{formatTime(liveTime)}</span>
        </div>

        <div className="status-badge" data-status={status.connected ? 'online' : 'offline'}>
          <Radio size={14} />
          <span>{status.connected ? 'System Online' : 'Disconnected'}</span>
        </div>
      </div>

      <div className="topbar-right">
        <button className="icon-btn" title="Refresh" onClick={() => window.location.reload()}>
          <RefreshCw size={20} />
        </button>

        <motion.button
          className="btn btn-danger manual-override"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <AlertTriangle size={18} />
          Manual Override
        </motion.button>
      </div>

      <style>{`
        .topbar {
          height: var(--topbar-height);
          background: var(--bg-secondary);
          border-bottom: 1px solid var(--border-primary);
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 24px;
        }

        .project-title {
          display: flex;
          flex-direction: column;
        }

        .project-title .gradient-text {
          font-size: 22px;
          font-weight: 800;
          letter-spacing: 2px;
        }

        .project-title .subtitle {
          font-size: 11px;
          color: var(--text-secondary);
          font-weight: 500;
          letter-spacing: 0.5px;
        }

        .topbar-center {
          display: flex;
          align-items: center;
          gap: 20px;
        }

        .location-badge {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          background: var(--bg-tertiary);
          border-radius: 8px;
          font-size: 13px;
          color: var(--text-secondary);
          min-width: 200px;
          justify-content: center;
        }

        .venue-display {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }

        .venue-display:hover .edit-icon {
          opacity: 1;
        }

        .edit-icon {
          opacity: 0.5;
          transition: opacity 0.2s;
        }

        .edit-venue-container {
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .venue-input {
          background: var(--bg-primary);
          border: 1px solid var(--border-primary);
          color: var(--text-primary);
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 12px;
          width: 150px;
        }

        .edit-btn {
          background: transparent;
          border: none;
          cursor: pointer;
          color: var(--text-secondary);
          padding: 2px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .edit-btn.save:hover { color: var(--status-safe); }
        .edit-btn.cancel:hover { color: var(--status-critical); }

        .live-badge {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          background: rgba(63, 185, 80, 0.15);
          border: 1px solid rgba(63, 185, 80, 0.3);
          border-radius: 8px;
          font-size: 13px;
          font-weight: 600;
          color: var(--status-safe);
        }

        .live-dot {
          width: 8px;
          height: 8px;
          background: var(--status-safe);
          border-radius: 50%;
          animation: pulse 1.5s ease-in-out infinite;
        }

        .live-time {
          font-family: 'SF Mono', 'Consolas', monospace;
          font-size: 12px;
        }

        .status-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 14px;
          background: var(--bg-tertiary);
          border-radius: 8px;
          font-size: 12px;
          color: var(--text-secondary);
        }

        .status-badge[data-status="online"] { color: var(--status-safe); }
        .status-badge[data-status="offline"] { color: var(--status-critical); }

        .topbar-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .icon-btn {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          border: 1px solid var(--border-primary);
          background: var(--bg-tertiary);
          color: var(--text-secondary);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
        }

        .icon-btn:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
          border-color: var(--accent-blue);
        }

        .manual-override {
          font-size: 13px;
          padding: 10px 18px;
        }
      `}</style>
    </header>
  );
}

// Spatial Map Component
function SpatialMap({ status, getDensityColor, isHeatmapMain, videoKey, videoError, setVideoError, onRefresh }) {
  const zones = [
    { id: 'A', x: 15, y: 20, width: 30, height: 35 },
    { id: 'B', x: 55, y: 20, width: 30, height: 35 },
    { id: 'C', x: 15, y: 60, width: 30, height: 30 },
    { id: 'D', x: 55, y: 60, width: 30, height: 30 }
  ];

  const getZoneCount = (zoneId) => {
    const zoneMap = {
      'A': (status.zones?.top_left || 0) + (status.zones?.top_center || 0),
      'B': (status.zones?.top_right || 0) + (status.zones?.mid_right || 0),
      'C': (status.zones?.mid_left || 0) + (status.zones?.bot_left || 0),
      'D': (status.zones?.mid_center || 0) + (status.zones?.bot_center || 0) + (status.zones?.bot_right || 0)
    };
    return zoneMap[zoneId] || 0;
  };

  const getZoneIntensity = (count) => {
    if (count > 30) return 0.8;
    if (count > 20) return 0.6;
    if (count > 10) return 0.4;
    if (count > 5) return 0.25;
    return 0.1;
  };

  return (
    <div className="spatial-map">
      {/* Video Feed */}
      <div className="video-background">
        {!videoError ? (
          <img
            key={videoKey}
            src={`http://localhost:8000/video-feed?t=${videoKey}`}
            alt="Live Feed"
            className="video-feed"
            onError={() => setVideoError(true)}
            onLoad={() => setVideoError(false)}
          />
        ) : (
          <div className="video-error">
            <Camera size={48} />
            <p>Video feed unavailable</p>
            <button onClick={onRefresh} className="btn-primary">
              <RefreshCw size={16} />
              Retry
            </button>
          </div>
        )}

        {/* Heatmap Overlay removed */}

        <div className="video-overlay-gradient" />
      </div>

      {/* Zone Labels removed */}

      {/* Incident Markers */}
      {status.fire_detected && (
        <motion.div
          className="incident-marker fire"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          style={{ left: '70%', top: '30%' }}
        >
          <Flame size={24} />
          <div className="pulse-ring" />
        </motion.div>
      )}

      {status.anomaly_detected && (
        <motion.div
          className="incident-marker anomaly"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          style={{ left: '35%', top: '45%' }}
        >
          <AlertTriangle size={20} />
          <div className="pulse-ring" />
        </motion.div>
      )}

      {/* Stats Overlay */}
      <div className="map-stats">
        <div className="stat-item">
          <Users size={18} />
          <div className="stat-content">
            <span className="stat-value">{status.person_count}</span>
            <span className="stat-label">People</span>
          </div>
        </div>
        <div className="stat-item" style={{ borderColor: getDensityColor(status.density_level) }}>
          <Activity size={18} style={{ color: getDensityColor(status.density_level) }} />
          <div className="stat-content">
            <span className="stat-value" style={{ color: getDensityColor(status.density_level) }}>
              {status.density_level}
            </span>
            <span className="stat-label">Density</span>
          </div>
        </div>
        <div className="stat-item">
          <Zap size={18} />
          <div className="stat-content">
            <span className="stat-value">{status.risk_score}</span>
            <span className="stat-label">Risk</span>
          </div>
        </div>
      </div>

      {/* View Mode Badge */}
      <div className="view-mode-badge">
        <Eye size={14} />
        <span>{isHeatmapMain ? 'Heatmap View' : 'Standard View'}</span>
      </div>

      <style>{`
        .spatial-map {
          position: relative;
          width: 100%;
          height: 100%;
          overflow: hidden;
          background: #0a0a0a;
        }

        .video-background {
          position: absolute;
          inset: 0;
        }

        .video-feed {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .video-error {
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 16px;
          color: var(--text-muted);
        }

        .video-error button {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .heatmap-overlay {
          position: absolute;
          inset: 0;
          pointer-events: none;
          mix-blend-mode: screen;
          opacity: 0.7;
        }

        .heatmap-svg {
          width: 100%;
          height: 100%;
        }

        .video-overlay-gradient {
          position: absolute;
          inset: 0;
          background: linear-gradient(
            180deg,
            rgba(11, 14, 20, 0.3) 0%,
            rgba(11, 14, 20, 0.05) 50%,
            rgba(11, 14, 20, 0.4) 100%
          );
          pointer-events: none;
        }

        .zone-label {
          position: absolute;
          transform: translate(-50%, -50%);
          background: rgba(11, 14, 20, 0.85);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 12px 18px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
          min-width: 80px;
          pointer-events: none;
        }

        .zone-name {
          font-size: 11px;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .zone-count {
          font-size: 28px;
          font-weight: 800;
        }

        .incident-marker {
          position: absolute;
          transform: translate(-50%, -50%);
          width: 56px;
          height: 56px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 10;
        }

        .incident-marker.fire {
          background: rgba(248, 81, 73, 0.25);
          color: var(--status-critical);
          border: 2px solid var(--status-critical);
          box-shadow: 0 0 30px rgba(248, 81, 73, 0.5);
        }

        .incident-marker.anomaly {
          background: rgba(210, 153, 34, 0.25);
          color: var(--status-warning);
          border: 2px solid var(--status-warning);
        }

        .pulse-ring {
          position: absolute;
          inset: -6px;
          border-radius: 50%;
          border: 2px solid currentColor;
          animation: pulse-ring 1.5s ease-out infinite;
        }

        @keyframes pulse-ring {
          0% { transform: scale(0.8); opacity: 1; }
          100% { transform: scale(2.2); opacity: 0; }
        }

        .map-stats {
          position: absolute;
          top: 20px;
          left: 20px;
          display: flex;
          gap: 12px;
        }

        .stat-item {
          background: rgba(11, 14, 20, 0.9);
          backdrop-filter: blur(10px);
          border: 1px solid var(--border-primary);
          border-radius: 12px;
          padding: 14px 18px;
          display: flex;
          align-items: center;
          gap: 12px;
          color: var(--text-primary);
        }

        .stat-content {
          display: flex;
          flex-direction: column;
        }

        .stat-value {
          font-size: 20px;
          font-weight: 700;
          line-height: 1;
        }

        .stat-label {
          font-size: 10px;
          color: var(--text-muted);
          margin-top: 2px;
          text-transform: uppercase;
        }

        .view-mode-badge {
          position: absolute;
          top: 20px;
          right: 20px;
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 16px;
          background: rgba(11, 14, 20, 0.9);
          backdrop-filter: blur(10px);
          border: 1px solid var(--border-primary);
          border-radius: 8px;
          font-size: 12px;
          color: var(--text-secondary);
        }
      `}</style>
    </div>
  );
}

// Video Preview Component (Bottom Left PIP)
function VideoPreview({ isHeatmapMain, onSwap, videoKey }) {
  return (
    <motion.div
      className="video-preview"
      initial={{ opacity: 0, x: -50 }}
      animate={{ opacity: 1, x: 0 }}
      onClick={onSwap}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="preview-header">
        <Camera size={12} />
        <span>{isHeatmapMain ? 'Raw Feed' : 'Heatmap'}</span>
        <div className="swap-hint">
          <RotateCcw size={10} />
          Swap
        </div>
      </div>
      <div className="preview-video">
        <img
          key={videoKey}
          src={`http://localhost:8000/video-feed?t=${videoKey}`}
          alt="Preview"
          onError={(e) => {
            e.target.style.display = 'none';
            e.target.nextSibling.style.display = 'flex';
          }}
        />
        <div className="preview-placeholder" style={{ display: 'none' }}>
          <Camera size={20} />
        </div>
        <div className="preview-overlay">
          <RotateCcw size={24} />
        </div>
      </div>

      <style>{`
        .video-preview {
          position: fixed;
          bottom: 120px;
          left: 90px;
          width: 220px;
          background: var(--bg-secondary);
          border: 1px solid var(--border-primary);
          border-radius: 12px;
          overflow: hidden;
          cursor: pointer;
          z-index: 100;
          transition: all 0.2s ease;
        }

        .video-preview:hover {
          border-color: var(--accent-blue);
          box-shadow: 0 0 20px rgba(88, 166, 255, 0.2);
        }

        .preview-header {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 12px;
          font-size: 11px;
          font-weight: 600;
          border-bottom: 1px solid var(--border-primary);
          color: var(--text-secondary);
        }

        .swap-hint {
          margin-left: auto;
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 9px;
          color: var(--accent-blue);
        }

        .preview-video {
          position: relative;
          height: 120px;
          background: #000;
        }

        .preview-video img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .preview-placeholder {
          width: 100%;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-muted);
        }

        .preview-overlay {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          opacity: 0;
          transition: opacity 0.2s ease;
        }

        .video-preview:hover .preview-overlay {
          opacity: 1;
        }
      `}</style>
    </motion.div>
  );
}

// Camera Switcher Component
function CameraSwitcher({ selectedCamera, onCameraSelect }) {
  const cameras = [
    { id: 'webcam', name: 'Main Camera', icon: Webcam, status: 'active' },
    { id: 'cam2', name: 'Entrance Gate', icon: Monitor, status: 'inactive' },
    { id: 'cam3', name: 'Exit Zone A', icon: Monitor, status: 'inactive' },
    { id: 'cam4', name: 'VIP Section', icon: Monitor, status: 'inactive' },
  ];

  return (
    <div className="camera-switcher">
      <div className="switcher-header">
        <Camera size={16} />
        <span>Camera Sources</span>
        <span className="camera-count">1 Active</span>
      </div>

      <div className="camera-list">
        {cameras.map(camera => {
          const Icon = camera.icon;
          return (
            <motion.button
              key={camera.id}
              className={`camera-card ${selectedCamera === camera.id ? 'active' : ''} ${camera.status}`}
              onClick={() => onCameraSelect(camera.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="camera-icon">
                <Icon size={20} />
              </div>
              <div className="camera-info">
                <span className="camera-name">{camera.name}</span>
                <span className="camera-type">{camera.id === 'webcam' ? 'USB Camera' : 'IP Stream'}</span>
              </div>
              <div className="camera-status">
                <div className={`status-indicator ${camera.status}`} />
                <span>{camera.status === 'active' ? 'Live' : 'Offline'}</span>
              </div>
              {selectedCamera === camera.id && (
                <motion.div className="selected-indicator" layoutId="selectedCamera" />
              )}
            </motion.button>
          );
        })}
      </div>

      <style>{`
        .camera-switcher {
          height: 100px;
          background: var(--bg-secondary);
          border-top: 1px solid var(--border-primary);
          padding: 12px 24px;
        }

        .switcher-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 12px;
          font-size: 13px;
          font-weight: 600;
          color: var(--text-secondary);
        }

        .camera-count {
          margin-left: auto;
          font-size: 11px;
          color: var(--status-safe);
          background: rgba(63, 185, 80, 0.15);
          padding: 4px 10px;
          border-radius: 12px;
        }

        .camera-list {
          display: flex;
          gap: 12px;
          overflow-x: auto;
        }

        .camera-card {
          position: relative;
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 16px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-primary);
          border-radius: 10px;
          cursor: pointer;
          transition: all 0.2s ease;
          min-width: 200px;
        }

        .camera-card:hover {
          background: var(--bg-hover);
          border-color: var(--accent-blue);
        }

        .camera-card.active {
          border-color: var(--accent-blue);
          background: rgba(88, 166, 255, 0.1);
        }

        .camera-card.inactive {
          opacity: 0.5;
        }

        .camera-icon {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          background: var(--bg-secondary);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-secondary);
        }

        .camera-card.active .camera-icon {
          background: var(--accent-blue);
          color: white;
        }

        .camera-info {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .camera-name {
          font-size: 13px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .camera-type {
          font-size: 10px;
          color: var(--text-muted);
        }

        .camera-status {
          margin-left: auto;
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 10px;
          color: var(--text-muted);
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .status-indicator.active {
          background: var(--status-safe);
          box-shadow: 0 0 8px rgba(63, 185, 80, 0.6);
        }

        .status-indicator.inactive {
          background: var(--text-muted);
        }

        .selected-indicator {
          position: absolute;
          bottom: -1px;
          left: 20%;
          right: 20%;
          height: 3px;
          background: var(--accent-blue);
          border-radius: 3px 3px 0 0;
        }
      `}</style>
    </div>
  );
}

// Alert Panel Component
function AlertPanel({ alerts, alertFilter, setAlertFilter }) {
  return (
    <aside className="alert-panel">
      <div className="panel-header">
        <h2>
          <Bell size={18} />
          Real-Time Alerts
        </h2>
        <span className="alert-count">{alerts.length}</span>
      </div>

      <div className="filter-tabs">
        {['all', 'critical', 'warning'].map(filter => (
          <button
            key={filter}
            className={`filter-tab ${alertFilter === filter ? 'active' : ''}`}
            onClick={() => setAlertFilter(filter)}
          >
            {filter.charAt(0).toUpperCase() + filter.slice(1)}
          </button>
        ))}
      </div>

      <div className="alerts-list">
        <AnimatePresence>
          {alerts.length === 0 ? (
            <div className="no-alerts">
              <CheckCircle size={40} />
              <p>No active alerts</p>
              <span>System operating normally</span>
            </div>
          ) : (
            alerts.map(alert => (
              <motion.div
                key={alert.id}
                className={`alert-card ${alert.type}`}
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
              >
                <div className="alert-icon">
                  {alert.type === 'critical' ? <AlertCircle size={20} /> : <AlertTriangle size={20} />}
                </div>
                <div className="alert-content">
                  <h4>{alert.title}</h4>
                  <p>{alert.message}</p>
                  <div className="alert-meta">
                    <span className="alert-time">{alert.time}</span>
                    <span className="alert-agent">{alert.agent}</span>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      <style>{`
        .alert-panel {
          width: var(--right-panel-width);
          background: var(--bg-secondary);
          border-radius: 16px;
          border: 1px solid var(--border-primary);
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .panel-header {
          padding: 20px;
          border-bottom: 1px solid var(--border-primary);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .panel-header h2 {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 16px;
          font-weight: 600;
        }

        .alert-count {
          background: var(--accent-blue);
          color: white;
          font-size: 12px;
          font-weight: 600;
          padding: 4px 10px;
          border-radius: 12px;
        }

        .filter-tabs {
          display: flex;
          padding: 12px 16px;
          gap: 8px;
          border-bottom: 1px solid var(--border-primary);
        }

        .filter-tab {
          flex: 1;
          padding: 8px 12px;
          border: none;
          border-radius: 8px;
          background: var(--bg-tertiary);
          color: var(--text-secondary);
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .filter-tab:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
        }

        .filter-tab.active {
          background: var(--accent-blue);
          color: white;
        }

        .alerts-list {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .no-alerts {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          color: var(--text-muted);
          gap: 8px;
        }

        .alert-card {
          background: var(--bg-tertiary);
          border-radius: 12px;
          padding: 16px;
          display: flex;
          gap: 14px;
          border-left: 4px solid;
        }

        .alert-card.critical {
          border-color: var(--status-critical);
          background: rgba(248, 81, 73, 0.08);
        }

        .alert-card.warning {
          border-color: var(--status-warning);
          background: rgba(210, 153, 34, 0.08);
        }

        .alert-icon {
          flex-shrink: 0;
        }

        .alert-card.critical .alert-icon {
          color: var(--status-critical);
        }

        .alert-card.warning .alert-icon {
          color: var(--status-warning);
        }

        .alert-content {
          flex: 1;
        }

        .alert-content h4 {
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 4px;
        }

        .alert-content p {
          font-size: 12px;
          color: var(--text-secondary);
          margin-bottom: 8px;
        }

        .alert-meta {
          display: flex;
          gap: 12px;
          font-size: 11px;
        }

        .alert-time {
          color: var(--text-muted);
        }

        .alert-agent {
          color: var(--accent-purple);
          background: rgba(163, 113, 247, 0.15);
          padding: 2px 8px;
          border-radius: 4px;
        }
      `}</style>
    </aside>
  );
}

// Glassmorphism Chat Component
function GlassmorphismChat({ isOpen, setIsOpen, isExpanded, setIsExpanded, messages, input, setInput, onSend, isLoading, status }) {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (!isOpen) return null;

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <motion.div
      className={`glass-chat ${isExpanded ? 'expanded' : ''}`}
      initial={{ opacity: 0, y: 50, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 50, scale: 0.9 }}
    >
      <div className="chat-header">
        <div className="chat-branding">
          <div className="gemini-logo">
            <Sparkles size={18} />
          </div>
          <div className="chat-title">
            <span className="title-main">Gemini LLM Agent</span>
            <span className="title-sub">Powered by AI</span>
          </div>
        </div>
        <div className="chat-controls">
          <button className="help-badge">
            <HelpCircle size={12} />
            Help
          </button>
          <button className="control-btn" onClick={() => setIsExpanded(!isExpanded)}>
            {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
          <button className="control-btn close" onClick={() => setIsOpen(false)}>
            <X size={16} />
          </button>
        </div>
      </div>

      <div className="context-bar">
        <div className="context-item">
          <Users size={12} />
          <span>{status.person_count} people</span>
        </div>
        <div className="context-item">
          <Activity size={12} />
          <span>{status.density_level}</span>
        </div>
        <div className="context-item">
          <Zap size={12} />
          <span>Risk: {status.risk_score}</span>
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            {msg.type === 'user' ? (
              <>
                <div className="message-content user-content">
                  {msg.text}
                </div>
                <div className="user-avatar">{msg.initials}</div>
              </>
            ) : (
              <>
                <div className="bot-avatar">
                  <Sparkles size={14} />
                </div>
                <div className="message-content bot-content">
                  <div className="message-text" style={{ color: msg.isError ? 'var(--status-critical)' : 'inherit' }}>
                    {msg.text}
                  </div>
                  {msg.agents && (
                    <div className="agent-attribution">
                      {msg.agents.map((agent, i) => (
                        <span key={i} className="agent-tag">{agent}</span>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="message bot">
            <div className="bot-avatar">
              <Sparkles size={14} />
            </div>
            <div className="message-content bot-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="quick-actions">
        <button onClick={() => setInput('How many people are in the venue?')}>Crowd count</button>
        <button onClick={() => setInput('What is the current risk level?')}>Risk analysis</button>
        <button onClick={() => setInput('Are there any active alerts?')}>Check alerts</button>
      </div>

      <div className="chat-input-area">
        <input
          type="text"
          placeholder="Ask a question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <button onClick={onSend} disabled={isLoading || !input.trim()}>
          <Send size={18} />
        </button>
      </div>

      <style>{`
        .glass-chat {
          position: fixed;
          bottom: 120px;
          right: 24px;
          width: 400px;
          height: 520px;
          background: rgba(13, 17, 23, 0.85);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 20px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          z-index: 2000;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .glass-chat.expanded {
          width: 500px;
          height: 650px;
        }

        .chat-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 20px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }

        .chat-branding {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .gemini-logo {
          width: 36px;
          height: 36px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }

        .chat-title {
          display: flex;
          flex-direction: column;
        }

        .title-main {
          font-size: 14px;
          font-weight: 600;
          color: #a5b4fc;
        }

        .title-sub {
          font-size: 10px;
          color: var(--text-muted);
        }

        .chat-controls {
          display: flex;
          gap: 8px;
        }

        .help-badge {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 6px 12px;
          background: rgba(163, 113, 247, 0.15);
          border: none;
          border-radius: 6px;
          color: var(--accent-purple);
          font-size: 11px;
          cursor: pointer;
        }

        .control-btn {
          width: 32px;
          height: 32px;
          border-radius: 8px;
          border: none;
          background: rgba(255, 255, 255, 0.05);
          color: var(--text-secondary);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .control-btn:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .context-bar {
          display: flex;
          gap: 16px;
          padding: 10px 20px;
          background: rgba(255, 255, 255, 0.02);
          border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .context-item {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          color: var(--text-muted);
        }

        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .message {
          display: flex;
          gap: 12px;
        }

        .message.user {
          flex-direction: row-reverse;
        }

        .user-avatar {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: 600;
          color: white;
        }

        .bot-avatar {
          width: 32px;
          height: 32px;
          border-radius: 10px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }

        .message-content {
          max-width: 80%;
        }

        .user-content {
          background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
          color: white;
          padding: 12px 16px;
          border-radius: 16px 16px 4px 16px;
          font-size: 13px;
        }

        .bot-content {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          padding: 14px 16px;
          border-radius: 16px 16px 16px 4px;
        }

        .message-text {
          font-size: 13px;
          line-height: 1.6;
          color: var(--text-primary);
        }

        .agent-attribution {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-top: 10px;
          padding-top: 10px;
          border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .agent-tag {
          font-size: 10px;
          padding: 3px 8px;
          background: rgba(163, 113, 247, 0.15);
          color: var(--accent-purple);
          border-radius: 4px;
        }

        .typing-indicator {
          display: flex;
          gap: 4px;
          padding: 8px 0;
        }

        .typing-indicator span {
          width: 8px;
          height: 8px;
          background: var(--text-muted);
          border-radius: 50%;
          animation: typing 1.4s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30% { transform: translateY(-6px); opacity: 1; }
        }

        .quick-actions {
          display: flex;
          gap: 8px;
          padding: 12px 20px;
          border-top: 1px solid rgba(255, 255, 255, 0.04);
          overflow-x: auto;
        }

        .quick-actions button {
          padding: 8px 14px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 20px;
          color: var(--text-secondary);
          font-size: 11px;
          cursor: pointer;
          white-space: nowrap;
        }

        .quick-actions button:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: var(--accent-blue);
        }

        .chat-input-area {
          display: flex;
          gap: 10px;
          padding: 16px 20px;
          border-top: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(0, 0, 0, 0.2);
        }

        .chat-input-area input {
          flex: 1;
          padding: 14px 18px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(255, 255, 255, 0.05);
          color: var(--text-primary);
          font-size: 13px;
          outline: none;
        }

        .chat-input-area input:focus {
          border-color: var(--accent-blue);
        }

        .chat-input-area button {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          border: none;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .chat-input-area button:hover:not(:disabled) {
          transform: scale(1.05);
        }

        .chat-input-area button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </motion.div>
  );
}

export default App;