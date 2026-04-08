from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str

class VideoSourceRequest(BaseModel):
    type: str  # 'webcam' or 'file'
    path: Optional[str] = None

class SystemConfigRequest(BaseModel):
    video_source_type: str  # 'webcam' or 'file'
    video_path: Optional[str] = None
    venue_name: str
    square_feet: float
