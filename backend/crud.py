"""
backend/crud.py
Database CRUD helper operations for Project Drishti
"""

import time
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.models_db import (
    Organization, User, Venue, CameraStream, AgentActuator,
    AnomalyIncident, AutonomousAction, AuditLog
)


def log_incident(
    db: Session,
    venue_id: str,
    threat_level: str,
    anomaly_code: str,
    crowd_count: int,
    crowd_density_per_m2: float,
    risk_score: int,
    fire_detected: bool = False,
    bounding_boxes_json: Optional[List[Dict[str, Any]]] = None,
    gemini_assessment: Optional[str] = None,
    camera_id: Optional[str] = None,
    agent_code: Optional[str] = "AnomalyAgent"
) -> AnomalyIncident:
    """Record an anomaly or hazard incident in the persistent database"""
    
    agent = None
    if agent_code:
        agent = db.query(AgentActuator).filter_by(agent_code=agent_code).first()

    incident = AnomalyIncident(
        venue_id=venue_id,
        camera_id=camera_id,
        detected_by_agent_id=agent.id if agent else None,
        threat_level=threat_level,
        anomaly_code=anomaly_code,
        crowd_count=crowd_count,
        crowd_density_per_m2=crowd_density_per_m2,
        risk_score=risk_score,
        fire_detected=fire_detected,
        bounding_boxes_json=bounding_boxes_json or [],
        gemini_assessment=gemini_assessment
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


def log_autonomous_action(
    db: Session,
    action_name: str,
    target_channel: str = "WEBHOOK_N8N",
    execution_status: str = "EXECUTED",
    incident_id: Optional[str] = None,
    payload_data: Optional[Dict[str, Any]] = None
) -> AutonomousAction:
    """Log an autonomous physical or network action executed by an agent"""
    
    action = AutonomousAction(
        incident_id=incident_id,
        action_name=action_name,
        target_channel=target_channel,
        execution_status=execution_status,
        payload_data=payload_data or {}
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def update_agent_stats(
    db: Session,
    agent_code: str,
    latency_ms: float
) -> Optional[AgentActuator]:
    """Update execution latency and increment invocation count for an agent actuator"""
    
    actuator = db.query(AgentActuator).filter_by(agent_code=agent_code).first()
    if actuator:
        actuator.last_latency_ms = float(latency_ms)
        actuator.invocation_count = (actuator.invocation_count or 0) + 1
        db.commit()
        db.refresh(actuator)
    return actuator


def get_all_agents(db: Session) -> List[AgentActuator]:
    """Fetch status and latency metrics for all 9 agent actuators"""
    return db.query(AgentActuator).all()


def get_recent_incidents(
    db: Session,
    limit: int = 50,
    severity: Optional[str] = None
) -> List[AnomalyIncident]:
    """Retrieve recent incidents sorted by timestamp descending"""
    query = db.query(AnomalyIncident)
    if severity and severity.upper() != "ALL":
        query = query.filter(AnomalyIncident.threat_level == severity.upper())
    return query.order_by(AnomalyIncident.created_at.desc()).limit(limit).all()
