from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class User(BaseModel):
    id: int
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str

class Note(BaseModel):
    id: Optional[int] = None
    title: str
    content: str
    summary: Optional[str] = None
    tags: Optional[str] = None
    type: str = "note"
    timestamp: Optional[datetime] = None
    user_id: Optional[int] = None
    status: str = "complete"

class Collection(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    color: str = "#4A90E2"
    icon: str = "📁"
    user_id: Optional[int] = None
