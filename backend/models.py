from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str

class VideoSourceRequest(BaseModel):
    type: str  # 'webcam' or 'file'
    path: Optional[str] = None
