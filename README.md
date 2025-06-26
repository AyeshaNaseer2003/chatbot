*Ayesha’s LangGraph + Tavily + Langfuse Chatbot*

A Python chatbot built with LangGraph, Gemini (via LangChain), TavilySearch, fastapi, pydantic model for api request and reponse, Langfuse tracing, and a MongoDB-powered history database. Deployed as a streaming Gradio app with chat-thread continuity.

🧱 Features
ReAct-style reasoning: Automatically triggers TavilySearch for current info (dates, weather, news).

Streaming responses: Character-by-character effect using Gradio.

MongoDB history: Saves each user/assistant exchange and retrieves past chats.

Langfuse observability: Traces prompts, tools used, and LLM generations.

Prompt versioning: Supports dynamic meta‑prompts from Langfuse for flexible control.