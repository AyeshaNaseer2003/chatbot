from typing import Annotated, TypedDict, List
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

# Langfuse imports
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

# Load env and API keys
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Setup Langfuse keys before initialization
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", os.getenv("LANGFUSE_PUBLIC_KEY"))
os.environ.setdefault("LANGFUSE_SECRET_KEY", os.getenv("LANGFUSE_SECRET_KEY"))
os.environ.setdefault("LANGFUSE_HOST", os.getenv("LANGFUSE_HOST"))
lf = get_client()
langfuse_handler = CallbackHandler()

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = init_chat_model("google_genai:gemini-2.0-flash")
tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

@observe()
def ask_gemini(message: str, history: list) -> str:
    # 1. Build system prompt
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant. Always use the Tavily search tool to answer questions needing current information, like dates or weather.")
    ]
    # 2. Inject history
    for pair in history:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            messages.append(HumanMessage(content=pair[0]))
            messages.append(AIMessage(content=pair[1]))
    # 3. Add current message
    messages.append(HumanMessage(content=message))

    # 4. Invoke Graph with Langfuse tracing callback
    output = graph.invoke(
        {"messages": messages},
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_user_id": os.getenv("USER", "unknown"),
                "langfuse_session_id": os.getenv("USER", "session"),
                "langfuse_tags": ["chatbot"]
            }
        }
    )

    # 5. Extract assistant reply safely
    final = output["messages"][-1]
    if isinstance(final, AIMessage):
        return final.content
    elif hasattr(final, "content"):
        return str(final.content)
    else:
        for msg in reversed(output["messages"]):
            if hasattr(msg, "content"):
                return str(msg.content)
    return "Sorry, I couldn’t generate a response."
