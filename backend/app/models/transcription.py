from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship

from backend.app.db.base import Base


class TranscriptionStatus(str, PyEnum):
    PENDING = "pending"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    PROCESSING_WITH_LLM = "processing_with_llm"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionType(str, PyEnum):
    SUMMARY = "summary"
    TECHNICAL_SPEC = "technical_spec"
    CUSTOM = "custom"


class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    transcription_type = Column(Enum(TranscriptionType), nullable=False)
    custom_prompt = Column(Text, nullable=True)
    status = Column(Enum(TranscriptionStatus), default=TranscriptionStatus.PENDING)
    error_message = Column(Text, nullable=True)
    raw_transcript = Column(Text, nullable=True)
    processed_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="transcriptions")
