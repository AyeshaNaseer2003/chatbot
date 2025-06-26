import time
import gradio as gr
import asyncio
from mongo_utils import save_message
from chatbot_agent import ask_gemini

user = "ayesha"

def sync_save(user_message, bot_reply):
    asyncio.run(save_message(user, user_message, bot_reply))

def slow_stream(user_message, history):
    history_pairs = [
        (m["content"], history[i+1]["content"])
        for i, m in enumerate(history[:-1])
        if m["role"] == "user"
    ]

    reply = ask_gemini(user_message, history_pairs)
    history = history + [{"role": "user", "content": user_message}]

    # Streaming tokens
    for i in range(len(reply)):
        current = reply[:i+1]
        yield history + [{"role": "assistant", "content": current}], current

    sync_save(user_message, reply)

demo = gr.ChatInterface(
    slow_stream,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
)

if __name__ == "__main__":
    demo.launch()
