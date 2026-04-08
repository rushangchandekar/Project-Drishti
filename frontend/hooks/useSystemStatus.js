import { useState, useEffect, useRef, useCallback } from 'react';

export function useSystemStatus(isSetupComplete) {
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
  const [agentFeed, setAgentFeed] = useState([]);
  const [autonomousActions, setAutonomousActions] = useState([]);
  
  // Track previous critical states via refs (not state) to avoid stale closures
  const prevFireRef = useRef(false);
  const prevDensityRef = useRef('UNKNOWN');
  const sosPlayingRef = useRef(false);

  // SOS Siren - plays a loud, unmistakable alarm pattern
  const playSOS = useCallback(() => {
    if (sosPlayingRef.current) return; // Don't stack multiple sirens
    sosPlayingRef.current = true;
    
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const masterGain = ctx.createGain();
      masterGain.gain.setValueAtTime(0.6, ctx.currentTime);
      masterGain.connect(ctx.destination);

      // Play 5 alternating high-low beeps (classic siren pattern)
      for (let i = 0; i < 5; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(masterGain);
        
        osc.type = 'square';
        const startTime = ctx.currentTime + i * 0.4;
        // Alternate between high and low pitch
        osc.frequency.setValueAtTime(i % 2 === 0 ? 880 : 1320, startTime);
        
        gain.gain.setValueAtTime(0.7, startTime);
        gain.gain.exponentialRampToValueAtTime(0.01, startTime + 0.35);
        
        osc.start(startTime);
        osc.stop(startTime + 0.35);
      }

      // Reset after siren finishes
      setTimeout(() => {
        sosPlayingRef.current = false;
        ctx.close().catch(() => {});
      }, 2500);
      
    } catch (e) {
      console.warn("SOS Audio failed:", e);
      sosPlayingRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (!isSetupComplete) return;

    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/status');
        const data = await response.json();

        // Detect transitions using refs (guaranteed fresh values)
        const fireJustStarted = data.fire_detected && !prevFireRef.current;
        const densityJustCritical = data.density_level === 'CRITICAL' && prevDensityRef.current !== 'CRITICAL';

        // Update refs immediately
        prevFireRef.current = data.fire_detected;
        prevDensityRef.current = data.density_level;

        // Generate alerts for new critical events
        const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        if (fireJustStarted) {
          setAlerts(prev => [{ id: Date.now(), type: 'critical', title: 'Fire Hazard Detected', message: 'Smoke/Fire detected by VisionAgent. Immediate attention required.', agent: 'FireAgent', time: currentTime }, ...prev].slice(0, 20));
          playSOS();
        }
        
        if (densityJustCritical) {
          setAlerts(prev => [{ id: Date.now() + 1, type: 'critical', title: 'Critical Density', message: 'Crowd density has reached critical levels.', agent: 'CrowdAgent', time: currentTime }, ...prev].slice(0, 20));
          playSOS();
        }
        
        if (data.anomaly_detected) {
          setAlerts(prev => {
            const hasRecent = prev.some(a => a.title === 'Anomaly Detected' && Date.now() - a.id < 10000);
            if (hasRecent) return prev;
            return [{ id: Date.now() + 2, type: 'warning', title: 'Anomaly Detected', message: data.anomaly_type || 'Unusual pattern detected.', agent: 'AnomalyAgent', time: currentTime }, ...prev].slice(0, 20);
          });
        }

        if (data.recent_agent_actions && data.recent_agent_actions.length > 0) {
          setAgentFeed(data.recent_agent_actions);
        }

        if (data.autonomous_actions) {
          setAutonomousActions(data.autonomous_actions);
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
  }, [isSetupComplete, playSOS]);

  return { status, alerts, agentFeed, autonomousActions, setAlerts };
}
