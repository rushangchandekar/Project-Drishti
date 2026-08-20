"""
backend/models_db.py
SQLAlchemy Database Models and Initializer for Project Drishti
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, JSON, ForeignKey, BigInteger
from sqlalchemy.orm import relationship
from backend.database import Base, engine, SessionLocal


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    venues = relationship("Venue", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(150), nullable=False)
    role = Column(String(50), nullable=False, default="COMMAND_OPERATOR")  # SUPER_ADMIN, SAFETY_DIRECTOR, COMMAND_OPERATOR, FIELD_OFFICER
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")


class Venue(Base):
    __tablename__ = "venues"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    square_feet = Column(Float, default=5000.0)
    area_m2 = Column(Float, default=464.5)
    spatial_sectors_3x3 = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="venues")
    camera_streams = relationship("CameraStream", back_populates="venue", cascade="all, delete-orphan")
    incidents = relationship("AnomalyIncident", back_populates="venue", cascade="all, delete-orphan")


class CameraStream(Base):
    __tablename__ = "camera_streams"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    venue_id = Column(String(36), ForeignKey("venues.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(150), nullable=False)
    source_type = Column(String(50), nullable=False, default="webcam")  # webcam, file, rtsp
    source_path = Column(String(500), nullable=False, default="0")
    assigned_agent_type = Column(String(50), nullable=False, default="VisionAgent")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    venue = relationship("Venue", back_populates="camera_streams")
    incidents = relationship("AnomalyIncident", back_populates="camera_stream")


class AgentActuator(Base):
    __tablename__ = "agent_actuators"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_code = Column(String(50), unique=True, nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)  # EMERGENCY_SAFETY, COGNITIVE_INTELLIGENCE, TACTICAL_OPERATIONS
    description = Column(Text, nullable=True)
    last_latency_ms = Column(Float, default=0.0)
    invocation_count = Column(BigInteger, default=0)
    last_active_at = Column(DateTime, default=datetime.utcnow)

    incidents = relationship("AnomalyIncident", back_populates="detected_by_agent")


class AnomalyIncident(Base):
    __tablename__ = "anomalies_incidents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    venue_id = Column(String(36), ForeignKey("venues.id", ondelete="CASCADE"), nullable=False)
    camera_id = Column(String(36), ForeignKey("camera_streams.id", ondelete="SET NULL"), nullable=True)
    detected_by_agent_id = Column(String(36), ForeignKey("agent_actuators.id", ondelete="SET NULL"), nullable=True)
    threat_level = Column(String(20), nullable=False, default="INFO")  # INFO, WARNING, CRITICAL
    anomaly_code = Column(String(100), nullable=False)  # SUDDEN_INFLUX, SUDDEN_DISPERSAL, FIRE_DETECTED
    crowd_count = Column(Integer, default=0)
    crowd_density_per_m2 = Column(Float, default=0.0)
    risk_score = Column(Integer, default=0)
    fire_detected = Column(Boolean, default=False)
    bounding_boxes_json = Column(JSON, default=list)
    gemini_assessment = Column(Text, nullable=True)
    snapshot_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    venue = relationship("Venue", back_populates="incidents")
    camera_stream = relationship("CameraStream", back_populates="incidents")
    detected_by_agent = relationship("AgentActuator", back_populates="incidents")
    actions = relationship("AutonomousAction", back_populates="incident", cascade="all, delete-orphan")


class AutonomousAction(Base):
    __tablename__ = "autonomous_actions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    incident_id = Column(String(36), ForeignKey("anomalies_incidents.id", ondelete="CASCADE"), nullable=True)
    action_name = Column(String(150), nullable=False)
    execution_status = Column(String(50), nullable=False, default="EXECUTED")  # EXECUTED, PENDING, FAILED
    target_channel = Column(String(50), nullable=False, default="WEBHOOK_N8N")
    payload_data = Column(JSON, default=dict)
    executed_at = Column(DateTime, default=datetime.utcnow)

    incident = relationship("AnomalyIncident", back_populates="actions")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")


def init_db():
    """Create all tables and seed initial agent actuators, org, venue, and default admin"""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # 1. Seed Organization
        org = db.query(Organization).filter_by(slug="drishti-command").first()
        if not org:
            org = Organization(name="Drishti Command Center", slug="drishti-command")
            db.add(org)
            db.commit()
            db.refresh(org)

        # 2. Seed Default Venue
        venue = db.query(Venue).filter_by(organization_id=org.id).first()
        if not venue:
            venue = Venue(
                organization_id=org.id,
                name="Main Entrance & Courtyard",
                square_feet=5000.0,
                area_m2=464.5,
                spatial_sectors_3x3={
                    "top_left": {"count": 0}, "top_center": {"count": 0}, "top_right": {"count": 0},
                    "mid_left": {"count": 0}, "mid_center": {"count": 0}, "mid_right": {"count": 0},
                    "bottom_left": {"count": 0}, "bottom_center": {"count": 0}, "bottom_right": {"count": 0}
                }
            )
            db.add(venue)
            db.commit()
            db.refresh(venue)

        # 3. Seed Default Camera Streams
        cam0 = db.query(CameraStream).filter_by(name="Main Gate (Webcam)").first()
        if not cam0:
            cam0 = CameraStream(
                venue_id=venue.id,
                name="Main Gate (Webcam)",
                source_type="webcam",
                source_path="0",
                assigned_agent_type="VisionAgent"
            )
            db.add(cam0)
            db.commit()

        # 4. Seed 9 Agent Actuators
        agents_seed = [
            ("FireAgent", "Fire Agent", "EMERGENCY_SAFETY", "Handles fire detection, sprinkler activation, fire station contact"),
            ("CrowdAgent", "Crowd Agent", "EMERGENCY_SAFETY", "Manages crowd density, gate control, crowd dispersal"),
            ("EvacAgent", "Evac Agent", "EMERGENCY_SAFETY", "Coordinates evacuation routes, emergency exits, PA announcements"),
            ("AnomalyAgent", "Anomaly Agent", "COGNITIVE_INTELLIGENCE", "Investigates unusual crowd behavior, suspicious activity"),
            ("ForecastAgent", "Forecast Agent", "COGNITIVE_INTELLIGENCE", "Predicts crowd trends, escalation risk, resource needs"),
            ("MedicAgent", "Medic Agent", "COGNITIVE_INTELLIGENCE", "Deploys medical teams for stampede, crush, or health emergencies"),
            ("DispatchAgent", "Dispatch Agent", "TACTICAL_OPERATIONS", "Dispatches security personnel, law enforcement escalation"),
            ("LLMAgent", "LLM Agent", "TACTICAL_OPERATIONS", "Generates natural language situation reports, executive summaries"),
            ("SecurityAgent", "Security Agent", "TACTICAL_OPERATIONS", "Monitors access points, perimeter breaches, physical security")
        ]

        for code, name, cat, desc in agents_seed:
            existing = db.query(AgentActuator).filter_by(agent_code=code).first()
            if not existing:
                actuator = AgentActuator(
                    agent_code=code,
                    agent_name=name,
                    category=cat,
                    description=desc,
                    last_latency_ms=800.0,
                    invocation_count=1
                )
                db.add(actuator)

        db.commit()

        # 5. Seed Default Admin User
        admin = db.query(User).filter_by(email="admin@drishti.ai").first()
        if not admin:
            from passlib.hash import bcrypt as bcrypt_hash
            admin = User(
                organization_id=org.id,
                email="admin@drishti.ai",
                password_hash=bcrypt_hash.hash("drishti2026"),
                full_name="Drishti Admin",
                role="SUPER_ADMIN"
            )
            db.add(admin)
            db.commit()
            print("[OK] Default admin user seeded (admin@drishti.ai / drishti2026)")

        print("[OK] Database initialized and all seed data applied!")

    except Exception as e:
        db.rollback()
        print(f"[WARN] Error seeding database: {e}")
    finally:
        db.close()
