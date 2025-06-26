from typing import Annotated, TypedDict, List, Dict
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

client = get_client()
langfuse_handler = CallbackHandler()

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

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
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You have a web searching tool Tavily(use it for current, today's and recent events)"
    "If you need current information (dates, weather, etc.), respond in ReAct format:\n"
    "Thought: Do I need a tool? Yes\n"
    "Action: TavilySearch\n"
    "Action Input: <your_question>\n"
    "Then wait for the tool's response before replying."
)
@observe(as_type="generation", name="ask_gemini")
def ask_gemini(message: str, history: List[List[str]]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, b in history:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": b}]
    messages.append({"role": "user", "content": message})

    with client.start_as_current_span(name="ask_gemini_graph"):
        output = graph.invoke(
            input={"messages": messages},
            config={"callbacks": [langfuse_handler]}
            # metadata={"langfuse_user_id": "ayesha"}
        )

    return output["messages"][-1].content

