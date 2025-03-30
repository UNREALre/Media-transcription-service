from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

from backend.app.models.transcription import TranscriptionStatus, TranscriptionType


# Shared properties
class TranscriptionBase(BaseModel):
    original_filename: Optional[str] = None
    transcription_type: Optional[TranscriptionType] = None
    custom_prompt: Optional[str] = None


# Properties to receive via API on creation
class TranscriptionCreate(TranscriptionBase):
    original_filename: str
    transcription_type: TranscriptionType
    custom_prompt: Optional[str] = None


# Properties to receive via API on update
class TranscriptionUpdate(BaseModel):
    status: Optional[TranscriptionStatus] = None
    error_message: Optional[str] = None
    raw_transcript: Optional[str] = None
    processed_text: Optional[str] = None
    completed_at: Optional[datetime] = None


# Properties shared by models stored in DB
class TranscriptionInDBBase(TranscriptionBase):
    id: int
    user_id: int
    file_path: str
    status: TranscriptionStatus
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    raw_transcript: Optional[str] = None
    processed_text: Optional[str] = None
    completed_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# Properties to return to client
class Transcription(TranscriptionInDBBase):
    pass


# Properties stored in DB
class TranscriptionInDB(TranscriptionInDBBase):
    pass


# For listing transcriptions
class TranscriptionList(BaseModel):
    transcriptions: List[Transcription]
    total: int
