import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import generate_response

load_dotenv()

app = FastAPI(
    title="Portfolio RAG Chatbot API",
    description="API for Lalitendra Swamy's Portfolio Chatbot"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_type: str
    chat_history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    suggested_questions: list[str]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Portfolio Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # Convert Pydantic models to dicts for the pipeline
        history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        response_data = generate_response(
            message=request.message,
            user_type=request.user_type,
            chat_history=history
        )
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
