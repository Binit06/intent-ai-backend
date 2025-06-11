from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import intent
from app.services.intent_service import load_bert_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(intent.router)

@app.on_event("startup")
async def startup_event():
    load_bert_model()
    print("FastAPI server is ready.")
