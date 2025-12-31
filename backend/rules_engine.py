"""
backend/rules_engine.py
Fast decision rules for immediate actions
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of actions that can be triggered"""
    FIRE_RESPONSE = "fire_response"
    CROWD_CONTROL = "crowd_control"
    MEDICAL_ALERT = "medical_alert"
    EVACUATION = "evacuation"
    MONITOR = "monitor"
    STANDBY = "standby"


class Priority(Enum):
    """Action priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Action:
    """Represents an action to be executed"""
    action_type: ActionType
    priority: Priority
    description: str
    webhook_endpoint: str
    payload: Dict[str, Any]
    immediate: bool = True  # Should execute immediately without LLM


class RulesEngine:
    """
    Fast, deterministic rules for emergency decisions
    No LLM delay - instant decisions for safety
    """
    
    def __init__(self, n8n_base_url: str):
        self.n8n_base_url = n8n_base_url
        
        # Define webhook endpoints
        self.webhooks = {
            "fire": f"{n8n_base_url}/fire-alert",
            "crowd": f"{n8n_base_url}/crowd-alert",
            "medical": f"{n8n_base_url}/medical-alert",
            "evacuation": f"{n8n_base_url}/evacuation",
        }
        
        print(f"[RULES ENGINE] Initialized with base URL: {n8n_base_url}")
    
    def evaluate(
        self,
        person_count: int,
        density_level: str,
        fire_detected: bool,
        anomaly_detected: bool,
        anomaly_type: str,
        trend: str,
        rate_of_change: float,
        risk_score: float,
        zones: Dict[str, int]
    ) -> List[Action]:
        """
        Evaluate current situation and return list of actions
        
        Returns:
            List of Action objects to execute
        """
        actions = []
        
        # RULE 1: Fire Detection (HIGHEST PRIORITY)
        if fire_detected:
            actions.append(Action(
                action_type=ActionType.FIRE_RESPONSE,
                priority=Priority.CRITICAL,
                description="Fire detected - Activate sprinklers and alert fire department",
                webhook_endpoint=self.webhooks["fire"],
                payload={
                    "event": "fire_detected",
                    "person_count": person_count,
                    "location": "main_area",
                    "timestamp": self._get_timestamp()
                },
                immediate=True
            ))
            
            # Also trigger evacuation
            actions.append(Action(
                action_type=ActionType.EVACUATION,
                priority=Priority.CRITICAL,
                description="Initiate emergency evacuation",
                webhook_endpoint=self.webhooks["evacuation"],
                payload={
                    "reason": "fire",
                    "person_count": person_count,
                    "timestamp": self._get_timestamp()
                },
                immediate=True
            ))
        
        # RULE 2: Critical Crowd Density
        if density_level == "CRITICAL":
            actions.append(Action(
                action_type=ActionType.CROWD_CONTROL,
                priority=Priority.CRITICAL,
                description="Critical crowd density - Deploy all response units",
                webhook_endpoint=self.webhooks["crowd"],
                payload={
                    "event": "critical_density",
                    "person_count": person_count,
                    "density": density_level,
                    "trend": trend,
                    "rate": rate_of_change,
                    "timestamp": self._get_timestamp()
                },
                immediate=True
            ))
        
        # RULE 3: High Density with Increasing Trend
        elif density_level == "HIGH" and trend == "INCREASING":
            actions.append(Action(
                action_type=ActionType.CROWD_CONTROL,
                priority=Priority.HIGH,
                description="High density increasing - Pre-position response teams",
                webhook_endpoint=self.webhooks["crowd"],
                payload={
                    "event": "high_density_increasing",
                    "person_count": person_count,
                    "density": density_level,
                    "trend": trend,
                    "rate": rate_of_change,
                    "timestamp": self._get_timestamp()
                },
                immediate=True
            ))
        
        # RULE 4: Sudden Influx (Panic potential)
        if anomaly_detected and anomaly_type == "SUDDEN_INFLUX":
            actions.append(Action(
                action_type=ActionType.CROWD_CONTROL,
                priority=Priority.HIGH,
                description="Sudden crowd influx detected - Monitor closely",
                webhook_endpoint=self.webhooks["crowd"],
                payload={
                    "event": "sudden_influx",
                    "person_count": person_count,
                    "rate": rate_of_change,
                    "timestamp": self._get_timestamp()
                },
                immediate=True
            ))
        
        # RULE 5: Zone Overcrowding
        if anomaly_detected and anomaly_type and "OVERCROWDING" in anomaly_type:
            zone = anomaly_type.replace("OVERCROWDING_IN_", "")
            actions.append(Action(
                action_type=ActionType.CROWD_CONTROL,
                priority=Priority.MEDIUM,
                description=f"Overcrowding in {zone} - Redirect crowd",
                webhook_endpoint=self.webhooks["crowd"],
                payload={
                    "event": "zone_overcrowding",
                    "zone": zone,
                    "person_count": person_count,
                    "zones": zones,
                    "timestamp": self._get_timestamp()
                },
                immediate=True
            ))
        
        # RULE 6: High Risk Score
        if risk_score >= 80:
            # Check if we haven't already added critical actions
            if not any(a.priority == Priority.CRITICAL for a in actions):
                actions.append(Action(
                    action_type=ActionType.CROWD_CONTROL,
                    priority=Priority.HIGH,
                    description=f"High risk score ({risk_score}) - Alert command center",
                    webhook_endpoint=self.webhooks["crowd"],
                    payload={
                        "event": "high_risk_score",
                        "risk_score": risk_score,
                        "person_count": person_count,
                        "density": density_level,
                        "timestamp": self._get_timestamp()
                    },
                    immediate=True
                ))
        
        # If no critical actions, return standby status
        if not actions:
            actions.append(Action(
                action_type=ActionType.STANDBY,
                priority=Priority.LOW,
                description="Situation normal - Continue monitoring",
                webhook_endpoint=None,
                payload={
                    "event": "normal",
                    "person_count": person_count,
                    "density": density_level,
                    "timestamp": self._get_timestamp()
                },
                immediate=False
            ))
        
        return actions
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()


# Test
if __name__ == "__main__":
    engine = RulesEngine(n8n_base_url="http://localhost:5678/webhook")
    
    # Test Case 1: Fire detected
    print("\n=== TEST 1: Fire Detected ===")
    actions = engine.evaluate(
        person_count=45,
        density_level="MEDIUM",
        fire_detected=True,
        anomaly_detected=True,
        anomaly_type="FIRE_DETECTED",
        trend="STABLE",
        rate_of_change=0.0,
        risk_score=95,
        zones={}
    )
    
    for action in actions:
        print(f"[{action.priority.value.upper()}] {action.description}")
        print(f"  → Webhook: {action.webhook_endpoint}")
        print(f"  → Payload: {action.payload}")
    
    # Test Case 2: Critical crowd
    print("\n=== TEST 2: Critical Crowd ===")
    actions = engine.evaluate(
        person_count=150,
        density_level="CRITICAL",
        fire_detected=False,
        anomaly_detected=False,
        anomaly_type=None,
        trend="INCREASING",
        rate_of_change=2.5,
        risk_score=85,
        zones={}
    )
    
    for action in actions:
        print(f"[{action.priority.value.upper()}] {action.description}")
    
    # Test Case 3: Normal situation
    print("\n=== TEST 3: Normal Situation ===")
    actions = engine.evaluate(
        person_count=25,
        density_level="LOW",
        fire_detected=False,
        anomaly_detected=False,
        anomaly_type=None,
        trend="STABLE",
        rate_of_change=0.1,
        risk_score=15,
        zones={}
    )
    
    for action in actions:
        print(f"[{action.priority.value.upper()}] {action.description}")