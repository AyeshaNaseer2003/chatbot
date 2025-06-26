# test.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import observe
from langfuse import get_client

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

# Initialize Langfuse client
client = get_client()

@observe(as_type="generation")
def ask_gemini(prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    resp = llm.invoke(prompt)
    # Optionally update token/cost metadata:
    # from langfuse.decorators import langfuse_context
    # langfuse_context.update_current_observation(usage=...)
    return resp.content

if __name__ == "__main__":
    print(ask_gemini("Explain OOP in one sentence."))
    client.flush()
