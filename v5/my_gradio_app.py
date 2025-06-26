import time
import gradio as gr
import asyncio
from mongo_utils import save_message
from chatbot_agent import ask_gemini

user = "ayesha"

def sync_save(user_message, bot_reply):
    asyncio.run(save_message(user, user_message, bot_reply))

def slow_stream(user_message, history):
    # Build history_pairs for context
    history_pairs = [
        (m["content"], history[i + 1]["content"])
        for i, m in enumerate(history[:-1])
        if m["role"] == "user"
    ]

    reply = ask_gemini(user_message, history_pairs)
    sync_save(user_message, reply)

    # Stream assistant response character by character
    partial = ""
    for ch in reply:
        partial += ch
        time.sleep(0.01)
        yield partial  # yield just the assistant's message

demo = gr.ChatInterface(
    slow_stream,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
)

if __name__ == "__main__":
    demo.launch()
