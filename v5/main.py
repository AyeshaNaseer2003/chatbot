from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot_agent import ask_gemini
from mongo_utils import save_message, get_history

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ChatRequest(BaseModel):
    user_id: str = "ayesha"
    message: str

class ChatResponse(BaseModel):
    reply: str

class HistoryResponse(BaseModel):
    history: list


# Routes

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    messages = await get_history(request.user_id)
    history = [[m["message"], m["response"]] for m in messages]

    reply = ask_gemini(request.message, history)
    await save_message(request.user_id, request.message, reply)

    return ChatResponse(reply=reply)


@app.get("/history/{user_id}", response_model=HistoryResponse)
async def fetch_history(user_id: str):
    messages = await get_history(user_id)
    return HistoryResponse(history=messages)
