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
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API Key")
    
    # N8N Webhooks
    N8N_WEBHOOK_BASE_URL: str = Field(
        default="http://localhost:5678/webhook",
        description="Base URL for n8n webhooks"
    )
    
    # Detection Settings
    DETECTION_CONFIDENCE: float = Field(
        default=0.5,
        description="YOLO detection confidence threshold"
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
        default="yolov8n.pt",
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
        
        # Only show first 10 chars of API key for security
        api_key_preview = settings.GEMINI_API_KEY[:10] + "..." if len(settings.GEMINI_API_KEY) > 10 else "TOO_SHORT"
        print(f"Gemini API Key: {api_key_preview}")
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}")
        print("\nMake sure your .env file in project root contains:")
        print("GEMINI_API_KEY=your_key_here")