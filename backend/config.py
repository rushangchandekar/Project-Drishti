"""
backend/config.py
Configuration management for Project Drishti (Updated for Pydantic V2)
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import os


class Settings(BaseSettings):
    """
    Application settings loaded from .env file
    """
    
    # API Keys
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API Key")
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API Key")
    OPENROUTER_MODEL: str = Field(default="meta-llama/llama-3.3-70b-instruct", description="OpenRouter Model")
    OPENROUTER_VISION_MODEL: str = Field(default="google/gemini-2.0-flash-001", description="OpenRouter Vision Model")
    
    # N8N Webhooks
    N8N_WEBHOOK_BASE_URL: str = Field(
        default="http://localhost:5678/webhook",
        description="Base URL for n8n webhooks"
    )

    # Twilio Configuration
    TWILIO_ACCOUNT_SID: str = Field(default="", description="Twilio Account SID")
    TWILIO_AUTH_TOKEN: str = Field(default="", description="Twilio Auth Token")
    TWILIO_FROM_NUMBER: str = Field(default="", description="Twilio Registered Sender Number")
    TWILIO_TO_NUMBER: str = Field(default="", description="Admin Receiver Number")
    
    # Detection Settings
    DETECTION_CONFIDENCE: float = Field(
        default=0.35,
        description="YOLO detection confidence threshold (lower = catches more, but more false positives)"
    )
    CROWD_THRESHOLD_WARNING: int = Field(
        default=50,
        description="Person count for WARNING level"
    )
    CROWD_THRESHOLD_CRITICAL: int = Field(
        default=100,
        description="Person count for CRITICAL level"
    )
    
    # YOLO Model
    YOLO_MODEL_PATH: str = Field(
        default="yolo11n.pt",
        description="Path to YOLO model file"
    )
    
    # Server Settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # Video Settings
    VIDEO_SOURCE: str = Field(
        default="0",
        description="Video source (0 for webcam, or file path)"
    )

    # YOLO Optimization Settings
    YOLO_INPUT_SIZE: int = Field(
        default=640,
        description="YOLO input resolution (lower = faster, 320/480/640)"
    )
    YOLO_HALF_PRECISION: bool = Field(
        default=False,
        description="Use FP16 half-precision inference (requires CUDA GPU)"
    )
    YOLO_MAX_DETECTIONS: int = Field(
        default=100,
        description="Maximum number of detections per frame"
    )
    YOLO_NMS_IOU: float = Field(
        default=0.45,
        description="NMS IoU threshold for suppressing overlapping boxes"
    )

    # Tracking Settings
    YOLO_TRACKER: str = Field(
        default="bytetrack.yaml",
        description="Ultralytics tracker config: 'bytetrack.yaml' (fast, motion-only) "
                     "or 'botsort.yaml' (slower, includes Re-ID — better for dense/occluded crowds)"
    )

    # Fire Detection Settings (consumed by AdvancedFireDetector)
    FIRE_MODE: str = Field(
        default="hybrid",
        description="Fire detection mode: 'color', 'motion', or 'hybrid'"
    )
    FIRE_COLOR_THRESHOLD: float = Field(
        default=0.008,
        description="Minimum fraction of frame pixels classified as fire-colored to trigger detection"
    )
    FIRE_BRIGHTNESS_MIN: int = Field(
        default=130,
        description="Minimum average brightness (0-255) for a region to be classified as fire"
    )
    FIRE_TEMPORAL_CONSISTENCY: float = Field(
        default=0.5,
        description="Fraction of recent frames that must show fire for hybrid mode to confirm"
    )
    FIRE_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6,
        description="Minimum confidence required to report a confirmed fire detection"
    )

    # Multi-Stream Settings
    MAX_CAMERA_STREAMS: int = Field(
        default=8,
        description="Maximum concurrent camera streams"
    )

    
    # Database & Authentication Settings
    DATABASE_URL: str = Field(
        default="sqlite:///./drishti.db",
        description="Database connection string (SQLite fallback or PostgreSQL)"
    )
    JWT_SECRET: str = Field(
        default="drishti_super_secret_jwt_key_2026",
        description="Secret key for JWT token encoding/decoding"
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT encoding algorithm"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="JWT access token expiration in minutes"
    )
    
    # Pydantic V2 configuration
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Create cached settings instance
    Using lru_cache ensures we only load .env once
    """
    return Settings()


# Test
if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    print(f"Looking for .env file in parent directory...")
    
    env_path = "../.env"
    
    if os.path.exists(env_path):
        print(f"✅ Found .env at: {os.path.abspath(env_path)}")
    else:
        print(f"❌ .env file not found!")
        print(f"Please create .env in project root: {os.path.abspath(env_path)}")
        exit(1)
    
    try:
        settings = get_settings()
        print("\n✅ Configuration loaded successfully!")
        print(f"N8N Base URL: {settings.N8N_WEBHOOK_BASE_URL}")
        print(f"Crowd Warning Threshold: {settings.CROWD_THRESHOLD_WARNING}")
        print(f"Detection Confidence: {settings.DETECTION_CONFIDENCE}")
        print(f"YOLO Tracker: {settings.YOLO_TRACKER}")
        print(f"Fire Mode: {settings.FIRE_MODE}")
        
        # Only show first 10 chars of API key for security
        api_key_preview = settings.GEMINI_API_KEY[:10] + "..." if len(settings.GEMINI_API_KEY) > 10 else "TOO_SHORT"
        print(f"Gemini API Key: {api_key_preview}")
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}")
        print("\nMake sure your .env file in project root contains:")
        print("GEMINI_API_KEY=your_key_here")