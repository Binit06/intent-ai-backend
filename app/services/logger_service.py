from app.db.models import MessageLog, Ticket
from app.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

async def log_message(prompt: str, intent: str, model: str, db: AsyncSession):
    log = MessageLog(prompt=prompt, intent=intent, model_used=model)
    db.add(log)
    await db.commit()

    if intent == "qa_complaint":
        ticket = Ticket(prompt=prompt, intent=intent)
        db.add(ticket)
        await db.commit()
