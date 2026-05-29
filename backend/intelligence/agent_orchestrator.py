"""
intelligence/agent_orchestrator.py
Intelligent Agent Orchestrator for Project Drishti

Combines deterministic rule-based triggers with Gemini Vision analysis
to decide which of the 9 agents to invoke for any given situation.
Gemini Vision is only used for critical situations (fire, overcrowding,
anomaly, stampede, medical emergencies).
"""

import asyncio
import time
import json
import cv2
import base64
import traceback
from typing import Dict, Any, List, Optional, Set

from backend.config import get_settings

settings = get_settings()


# =============================================================================
# AGENT REGISTRY — All 9 Agents
# =============================================================================

AGENT_REGISTRY = {
    # ── Emergency Category ──
    "FireAgent": {
        "name": "Fire Agent",
        "category": "emergency",
        "webhook_path": "agent-fire",
        "description": "Handles fire detection, sprinkler activation, fire station contact",
        "icon": "flame",
    },
    "CrowdAgent": {
        "name": "Crowd Agent",
        "category": "emergency",
        "webhook_path": "agent-crowd",
        "description": "Manages crowd density, gate control, crowd dispersal",
        "icon": "users",
    },
    "EvacAgent": {
        "name": "Evac Agent",
        "category": "emergency",
        "webhook_path": "agent-evac",
        "description": "Coordinates evacuation routes, emergency exits, PA announcements",
        "icon": "log-out",
    },

    # ── Intelligence Category ──
    "AnomalyAgent": {
        "name": "Anomaly Agent",
        "category": "intelligence",
        "webhook_path": "agent-anomaly",
        "description": "Investigates unusual crowd behavior, suspicious activity",
        "icon": "alert-triangle",
    },
    "ForecastAgent": {
        "name": "Forecast Agent",
        "category": "intelligence",
        "webhook_path": "agent-forecast",
        "description": "Predicts crowd trends, escalation risk, resource needs",
        "icon": "trending-up",
    },
    "MedicAgent": {
        "name": "Medic Agent",
        "category": "intelligence",
        "webhook_path": "agent-medic",
        "description": "Deploys medical teams for stampede, crush, or health emergencies",
        "icon": "heart-pulse",
    },

    # ── Operations Category ──
    "DispatchAgent": {
        "name": "Dispatch Agent",
        "category": "operations",
        "webhook_path": "agent-dispatch",
        "description": "Dispatches security, staff, or emergency services to zones",
        "icon": "radio",
    },
    "LLMAgent": {
        "name": "LLM Agent",
        "category": "operations",
        "webhook_path": "agent-llm",
        "description": "Generates situation summaries, incident reports, command briefs",
        "icon": "brain",
    },
    "SecurityAgent": {
        "name": "Security Agent",
        "category": "operations",
        "webhook_path": "agent-security",
        "description": "Monitors perimeter, suspicious behavior, access control",
        "icon": "shield",
    },
}


def _build_initial_agent_statuses() -> Dict[str, Dict[str, Any]]:
    """Build initial status dict for all 9 agents."""
    statuses = {}
    for agent_id, info in AGENT_REGISTRY.items():
        statuses[agent_id] = {
            "agent_id": agent_id,
            "name": info["name"],
            "category": info["category"],
            "icon": info["icon"],
            "description": info["description"],
            "status": "idle",            # idle | running | completed | error
            "last_result": None,         # JSON result from n8n
            "last_error": None,
            "last_invoked": None,        # ISO timestamp
            "last_completed": None,
            "execution_time_ms": 0,
            "invocation_count": 0,
            "trigger_reason": None,      # Why this agent was activated
        }
    return statuses


class AgentOrchestrator:
    """
    Intelligent orchestrator that decides which agents to activate.

    Two-tier decision making:
    1. RULES (always): Fast deterministic triggers from detection data.
    2. GEMINI VISION (critical only): Sends frame screenshot + context
       to Gemini for deeper analysis when critical situations are detected.
    """

    def __init__(self, gemini_client=None, model_name="gemini-2.5-flash"):
        self.gemini_client = gemini_client
        self.model_name = model_name

        # Cooldown for Gemini Vision calls (seconds)
        self.vision_cooldown = 30
        self._last_vision_call = 0

        # Orchestration history
        self.history = []
        self.max_history = 50

        print("[AGENT ORCHESTRATOR] Initialized with 9 agents")

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def rule_based_selection(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Fast deterministic agent selection based on detection results.
        Returns dict of {agent_id: trigger_reason}.
        """
        selected: Dict[str, str] = {}

        fire = context.get("fire_detected", False)
        fire_conf = context.get("fire_confidence", 0)
        density = context.get("density_level", "UNKNOWN")
        risk = context.get("risk_score", 0)
        anomaly = context.get("anomaly_detected", False)
        anomaly_type = context.get("anomaly_type")
        anomaly_severity = context.get("anomaly_severity", "LOW")
        trend = context.get("trend", "STABLE")
        person_count = context.get("person_count", 0)

        # ── Emergency agents ──
        if fire:
            selected["FireAgent"] = f"Fire detected (confidence {fire_conf:.0%})"
            selected["EvacAgent"] = "Evacuation required due to fire"
            selected["DispatchAgent"] = "Deploy fire response resources"

        if density in ("CRITICAL", "VERY_HIGH"):
            selected["CrowdAgent"] = f"Crowd density {density}"
            if density == "CRITICAL":
                selected["EvacAgent"] = "Critical density - evacuation may be needed"
                selected["DispatchAgent"] = "Deploy crowd control security"

        # ── Intelligence agents ──
        if anomaly:
            selected["AnomalyAgent"] = f"Anomaly: {anomaly_type or 'unknown'}"
            if anomaly_type and "stampede" in str(anomaly_type).lower():
                selected["MedicAgent"] = "Stampede risk - medical standby"
                selected["EvacAgent"] = "Stampede risk - prepare evacuation"
            if anomaly_severity in ("CRITICAL", "HIGH"):
                selected["SecurityAgent"] = f"High-severity anomaly: {anomaly_type}"

        if risk > 60 or trend == "INCREASING":
            selected["ForecastAgent"] = f"Risk {risk:.0f}/100, trend {trend}"

        if risk > 80:
            selected["LLMAgent"] = f"High risk ({risk:.0f}) - generate situation report"
            selected["DispatchAgent"] = "High risk - resource deployment needed"

        # ── Medic for crush scenarios ──
        if density == "CRITICAL" and person_count > 80:
            selected["MedicAgent"] = "Crush risk - high density + high count"

        return selected

    def is_critical_situation(self, context: Dict[str, Any]) -> bool:
        """Check if the situation warrants Gemini Vision analysis."""
        return (
            context.get("fire_detected", False)
            or context.get("density_level") in ("CRITICAL", "VERY_HIGH")
            or context.get("anomaly_detected", False)
            or context.get("risk_score", 0) > 70
        )

    def can_call_vision(self) -> bool:
        """Check if we can make a Gemini Vision call (respecting cooldown)."""
        return (
            self.gemini_client is not None
            and (time.time() - self._last_vision_call) >= self.vision_cooldown
        )

    def gemini_vision_selection(
        self, context: Dict[str, Any], frame
    ) -> Dict[str, str]:
        """
        Send frame screenshot + context to Gemini Vision for intelligent
        agent selection. Only called for critical situations.

        Returns dict of {agent_id: trigger_reason}.
        """
        if not self.gemini_client or frame is None:
            return {}

        self._last_vision_call = time.time()

        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_bytes = buffer.tobytes()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Build context summary (no frame)
            ctx_summary = {
                "person_count": context.get("person_count", 0),
                "density_level": context.get("density_level", "UNKNOWN"),
                "risk_score": context.get("risk_score", 0),
                "fire_detected": context.get("fire_detected", False),
                "fire_confidence": context.get("fire_confidence", 0),
                "anomaly_detected": context.get("anomaly_detected", False),
                "anomaly_type": context.get("anomaly_type"),
                "anomaly_severity": context.get("anomaly_severity"),
                "trend": context.get("trend", "STABLE"),
                "zones": context.get("zones", {}),
                "situation_severity": context.get("situation_severity", "UNKNOWN"),
            }

            agent_names = list(AGENT_REGISTRY.keys())

            prompt = f"""You are the Agent Orchestrator for Project Drishti, an AI-powered crowd safety surveillance system.

You are analyzing a live surveillance camera frame along with detection data to decide which AI agents should be activated.

DETECTION DATA:
{json.dumps(ctx_summary, indent=2)}

AVAILABLE AGENTS (pick from these EXACT IDs):
{json.dumps({k: v['description'] for k, v in AGENT_REGISTRY.items()}, indent=2)}

INSTRUCTIONS:
1. Analyze the surveillance image AND the detection data together.
2. Decide which agents should be activated to handle this situation.
3. For each selected agent, provide a brief instruction/reason.
4. Only select agents that are genuinely needed. Don't activate agents for normal situations.
5. If the situation is truly normal despite high detection values, say so.

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
    "analysis": "Brief 1-2 sentence analysis of what you see in the frame",
    "agents": {{
        "AgentId": "instruction/reason for this agent"
    }}
}}

If no agents are needed, return: {{"analysis": "...", "agents": {{}}}}"""

            # Call Gemini with image
            from google.genai import types

            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=image_bytes,
                                )
                            ),
                            types.Part(text=prompt),
                        ]
                    )
                ],
            )

            text = response.text.strip()

            # Parse JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())
            analysis = result.get("analysis", "")
            agents = result.get("agents", {})

            print(f"\n[ORCHESTRATOR VISION] Analysis: {analysis}")

            # Validate agent IDs
            valid_agents = {}
            for agent_id, reason in agents.items():
                if agent_id in AGENT_REGISTRY:
                    valid_agents[agent_id] = f"[AI] {reason}"

            return valid_agents

        except Exception as e:
            print(f"[ORCHESTRATOR VISION] Gemini Vision failed: {e}")
            return {}

    def merge_selections(
        self, rule_agents: Dict[str, str], ai_agents: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Merge rule-based and AI-based agent selections.
        Rule-based agents are always included. AI can add extras.
        """
        merged = dict(rule_agents)
        for agent_id, reason in ai_agents.items():
            if agent_id not in merged:
                merged[agent_id] = reason
            else:
                # Append AI reasoning to existing rule reason
                merged[agent_id] = f"{merged[agent_id]} | {reason}"
        return merged

    def log_orchestration(
        self, selected_agents: Dict[str, str], method: str, context_summary: Dict
    ):
        """Log an orchestration decision."""
        entry = {
            "timestamp": time.time(),
            "method": method,
            "agents_selected": list(selected_agents.keys()),
            "agent_count": len(selected_agents),
            "context": {
                "person_count": context_summary.get("person_count", 0),
                "risk_score": context_summary.get("risk_score", 0),
                "density_level": context_summary.get("density_level", "UNKNOWN"),
                "fire_detected": context_summary.get("fire_detected", False),
                "anomaly_detected": context_summary.get("anomaly_detected", False),
            },
        }
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)
