from typing import Annotated, TypedDict, List
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

from langfuse import get_client,observe
from langfuse.langchain import CallbackHandler

lf = get_client()
lf_handler = CallbackHandler()

# ✔️ Fetch the production meta‑prompt
prompt_obj = lf.get_prompt("my_gemini_meta", type="chat")  # defaults to production label
SYSTEM_PROMPT_TEMPLATES = prompt_obj  # Prompt object, includes .prompt (list of messages) & .config

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

@observe(as_type="generation", name="ask_gemini")
def ask_gemini(message: str, history: List[List[str]]) -> str:
    # ✅ Compile prompt with current user_input
    system_messages = SYSTEM_PROMPT_TEMPLATES.compile(user_input=message)
    messages = system_messages.copy()  # list of role/content dicts

    # Append conversation history
    for u, b in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})

    # Optionally, not needed since compile appended user input
    # messages.append({"role":"user", "content":message})

    with lf.start_as_current_span(name="ask_gemini_graph"):
        output = graph.invoke(
            input={"messages": messages},
            config={
              "callbacks": [lf_handler],
              "langfuse_prompt": SYSTEM_PROMPT_TEMPLATES  # link prompt to trace
            },
            metadata={"langfuse_user_id": "ayesha"}
        )

    return output["messages"][-1].content
