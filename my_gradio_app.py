# my_gradio_app.py
import time
import gradio as gr
import asyncio
from mongo_utils import save_message
from chatbot_agent import ask_gemini  

user = "ayesha"

def sync_save(user_message, bot_reply):
    asyncio.run(save_message(user, user_message, bot_reply))

def slow_stream(user_message, history):

    response = ask_gemini(user_message, [(m["content"], history[i+1]["content"])
                                         for i, m in enumerate(history[:-1])
                                         if m["role"] == "user"])
    # Stream token by token
    partial = ""
    for token in response:
        partial += token
        time.sleep(0.01)
        yield partial  # yields just the partial content string
    
    # Save once completed
    sync_save(user_message, response)

# Use ChatInterface with streaming and DB saving
demo = gr.ChatInterface(
    fn=slow_stream,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True  
)

if __name__ == "__main__":
    demo.launch()
