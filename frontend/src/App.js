import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';

// Components
import TopBar from './components/TopBar';
import SpatialMap from './components/SpatialMap';
import VideoPreview from './components/VideoPreview';
import CameraSwitcher from './components/CameraSwitcher';
import AlertPanel from './components/AlertPanel';
import GlassmorphismChat from './components/GlassmorphismChat';

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
    setVideoKey(Date.now());

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

export default App;