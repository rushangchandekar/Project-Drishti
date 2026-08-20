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
  const [agentStatuses, setAgentStatuses] = useState({});
  
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

    let ws = null;
    let fallbackInterval = null;

    const handleData = (data) => {
      // Detect transitions using refs
      const fireJustStarted = data.fire_detected && !prevFireRef.current;
      const densityJustCritical = data.density_level === 'CRITICAL' && prevDensityRef.current !== 'CRITICAL';

      prevFireRef.current = data.fire_detected;
      prevDensityRef.current = data.density_level;

      const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const newAlerts = [];

      // 1. FireAgent
      if (fireJustStarted || data.fire_detected) {
        newAlerts.push({
          id: 'fire_' + Math.floor(Date.now() / 15000),
          type: 'critical',
          title: 'Fire Hazard Detected',
          message: `Smoke/Fire detected with ${Math.round((data.fire_confidence || 0.85) * 100)}% confidence. Fire suppression active.`,
          agent: 'FireAgent',
          time: currentTime
        });
        if (fireJustStarted) playSOS();
      }

      // 2. CrowdAgent
      if (densityJustCritical || data.density_level === 'CRITICAL' || data.density_level === 'HIGH' || data.density_level === 'VERY_HIGH') {
        const isCrit = data.density_level === 'CRITICAL' || data.density_level === 'VERY_HIGH';
        newAlerts.push({
          id: 'crowd_' + Math.floor(Date.now() / 15000),
          type: isCrit ? 'critical' : 'warning',
          title: isCrit ? 'Critical Crowd Density' : 'Elevated Crowd Surge',
          message: `Crowd count reached ${data.person_count || 0} people (${data.density_level}). Gate controls & monitoring active.`,
          agent: 'CrowdAgent',
          time: currentTime
        });
        if (densityJustCritical) playSOS();
      }

      // 3. AnomalyAgent
      if (data.anomaly_detected) {
        newAlerts.push({
          id: 'anomaly_' + Math.floor(Date.now() / 15000),
          type: data.anomaly_severity === 'CRITICAL' ? 'critical' : 'warning',
          title: 'Spatial Anomaly Detected',
          message: data.anomaly_type ? `Anomaly Code: ${data.anomaly_type}` : 'Unusual crowd movement pattern identified in spatial sector.',
          agent: 'AnomalyAgent',
          time: currentTime
        });
      }

      // 4. EvacAgent (Evacuation / Exit Doors / PA System)
      if (data.fire_detected || data.density_level === 'CRITICAL' || (data.activities && data.activities.some(a => a.type === 'PANIC' || a.type === 'STAMPEDE'))) {
        newAlerts.push({
          id: 'evac_' + Math.floor(Date.now() / 15000),
          type: 'critical',
          title: 'Evacuation Protocol Active',
          message: 'Opening emergency exit doors and activating public address announcements.',
          agent: 'EvacAgent',
          time: currentTime
        });
      }

      // 5. MedicAgent (Falls / Stampede / Health Emergencies)
      if (data.activities && data.activities.some(a => a.type === 'FALL' || a.type === 'STAMPEDE')) {
        const fallAct = data.activities.find(a => a.type === 'FALL' || a.type === 'STAMPEDE');
        newAlerts.push({
          id: 'medic_' + Math.floor(Date.now() / 15000),
          type: 'critical',
          title: 'Medical Emergency Alert',
          message: fallAct ? `Medical alert: ${fallAct.description || 'Fall / Injury detected in zone'}` : 'Paramedic unit dispatched to sector.',
          agent: 'MedicAgent',
          time: currentTime
        });
      }

      // 6. DispatchAgent (Fight / Security Dispatch)
      if ((data.activities && data.activities.some(a => a.type === 'FIGHT')) || (data.autonomous_actions && data.autonomous_actions.includes('Dispatching Security Personnel'))) {
        newAlerts.push({
          id: 'dispatch_' + Math.floor(Date.now() / 15000),
          type: 'critical',
          title: 'Tactical Security Dispatch',
          message: 'Dispatching rapid response security personnel to active incident zone.',
          agent: 'DispatchAgent',
          time: currentTime
        });
      }

      // 7. ForecastAgent (Escalation / Surge Prediction)
      if (data.trend === 'INCREASING' || (data.risk_score && data.risk_score > 55)) {
        newAlerts.push({
          id: 'forecast_' + Math.floor(Date.now() / 20000),
          type: 'warning',
          title: 'Crowd Escalation Forecast',
          message: `Surge trend detected (+${data.rate_of_change || 5.0} rate). Risk Score: ${Math.round(data.risk_score || 60)}/100.`,
          agent: 'ForecastAgent',
          time: currentTime
        });
      }

      // 8. SecurityAgent (Perimeter / High Severity)
      if (data.anomaly_severity === 'HIGH' || data.anomaly_severity === 'CRITICAL' || (data.risk_score && data.risk_score > 75)) {
        newAlerts.push({
          id: 'security_' + Math.floor(Date.now() / 20000),
          type: 'warning',
          title: 'Perimeter Security Alert',
          message: 'Access control and perimeter breach monitoring active across sectors.',
          agent: 'SecurityAgent',
          time: currentTime
        });
      }

      // 9. LLMAgent (Strategic Guidance / Situation Assessment)
      if (data.strategic_guidance && data.strategic_guidance !== 'NORMAL OPERATIONS: Continue routine surveillance.') {
        newAlerts.push({
          id: 'llm_' + Math.floor(Date.now() / 25000),
          type: 'warning',
          title: 'Strategic AI Briefing',
          message: data.strategic_guidance,
          agent: 'LLMAgent',
          time: currentTime
        });
      }

      // 10. Direct Selected Agents from Orchestrator
      if (data.selected_agents && typeof data.selected_agents === 'object') {
        Object.entries(data.selected_agents).forEach(([agentCode, reason]) => {
          newAlerts.push({
            id: `orch_${agentCode}_` + Math.floor(Date.now() / 15000),
            type: reason.toLowerCase().includes('fire') || reason.toLowerCase().includes('critical') ? 'critical' : 'warning',
            title: `${agentCode} Triggered`,
            message: reason,
            agent: agentCode,
            time: currentTime
          });
        });
      }

      if (newAlerts.length > 0) {
        setAlerts(prev => {
          const merged = [...newAlerts];
          prev.forEach(p => {
            if (!merged.some(m => m.id === p.id)) {
              merged.push(p);
            }
          });
          return merged.slice(0, 35);
        });
      }

      if (data.recent_agent_actions && data.recent_agent_actions.length > 0) {
        setAgentFeed(data.recent_agent_actions);
      }

      if (data.autonomous_actions) {
        setAutonomousActions(data.autonomous_actions);
      }

      setStatus({ ...data, connected: true });
    };

    const fetchAgentStatuses = async () => {
      try {
        const res = await fetch('http://localhost:8000/agent-statuses');
        if (res.ok) {
          const agentData = await res.json();
          setAgentStatuses(agentData);
        }
      } catch (err) {
        // silent catch
      }
    };

    const connectWebSocket = () => {
      try {
        const host = typeof window !== 'undefined' ? window.location.hostname || 'localhost' : 'localhost';
        ws = new WebSocket(`ws://${host}:8000/ws/telemetry`);

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleData(data);
          } catch (e) {
            console.error('WS parse error:', e);
          }
        };

        ws.onopen = () => {
          console.log('[DRISHTI WS] Real-time telemetry WebSocket connected');
          if (fallbackInterval) clearInterval(fallbackInterval);
        };

        ws.onerror = ws.onclose = () => {
          console.warn('[DRISHTI WS] Connection closed, falling back to HTTP polling');
          startPollingFallback();
        };
      } catch (e) {
        startPollingFallback();
      }
    };

    const startPollingFallback = () => {
      if (fallbackInterval) return;
      fallbackInterval = setInterval(async () => {
        try {
          const res = await fetch('http://localhost:8000/status');
          if (res.ok) {
            const data = await res.json();
            handleData(data);
          }
        } catch (err) {
          setStatus(prev => ({ ...prev, connected: false }));
        }
      }, 2000);
    };

    connectWebSocket();
    fetchAgentStatuses();
    const agentInterval = setInterval(fetchAgentStatuses, 3000);

    return () => {
      if (ws) ws.close();
      if (fallbackInterval) clearInterval(fallbackInterval);
      clearInterval(agentInterval);
    };
  }, [isSetupComplete, playSOS]);

  return { status, alerts, agentFeed, autonomousActions, agentStatuses, setAlerts };
}
