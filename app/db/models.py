from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from app.db.database import Base

class MessageLog(Base):
    __tablename__ = "message_logs"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    intent = Column(String, nullable=False)
    model_used = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    intent = Column(String, default="qa_complaint")
    status = Column(String, default="open")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
