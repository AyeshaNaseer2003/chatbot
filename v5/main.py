from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default_user")
    message = data.get("message")

    messages = await get_history(user_id)
    history = [[m["message"], m["response"]] for m in messages]

    reply = ask_gemini(message, history)

    await save_message(user_id, message, reply)

    return {"reply": reply}


@app.get("/history/{user_id}")
async def fetch_history(user_id: str):
    messages = await get_history(user_id)
    return {"history": messages}
