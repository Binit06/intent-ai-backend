from fastapi import APIRouter, HTTPException
from fastapi import Depends
from app.models.prompt_input import PromptInput
from app.db.database import get_db
from app.services.intent_service import predict_intent, client
from app.services.logger_service import log_message
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

@router.get("/")
async def home():
    return {"message": "Welcome to the Home route"}

@router.post("/generate-intent")
async def generate_text(prompt_input: PromptInput, db: AsyncSession = Depends(get_db)):
    try:
        user_prompt = prompt_input.prompt
        print(f"Received prompt: '{user_prompt}'")

        final_prompt = (
            "Categorize this in this category in 2 words only question related to credit cards and debit cards "
            "are considered accounts questions for the faws any theoretical question that can have the same answer "
            "for all user is considered faq question and anything other than these are complaint questions. "
            "Also no formatting on the text just plain text: (FAQ Question), (Complaint Question), "
            "(Account Question), (Greetings Question) : "
        ) + user_prompt

        intent = predict_intent(user_prompt)
        response_model = "bert"

        response = client.models.generate_content(model="gemini-2.0-flash", contents=final_prompt)
        text = response.text.lower()

        if "complaint" in text:
            intent = "qa_complaint"
            response_model = "gemini"

        if "greetings" in text:
            intent = "qa_greetings"
            response_model = "gemini"

        await log_message(user_prompt, intent, response_model, db=db)
        return {"prompt": user_prompt, "ai_response": intent}
    except Exception as e:
        print("Error in /generate-intent:", e)
        raise HTTPException(status_code=500, detail=str(e))
