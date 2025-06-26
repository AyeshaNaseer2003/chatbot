from typing import Annotated, TypedDict, List, Dict
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

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
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


def ask_gemini(message: str, history: list) -> str:
    # Build initial system prompt
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant. Always use the Tavily search tool to answer questions needing current information, like dates or weather.")
    ]

    # Add historical user-assistant pairs
    for pair in history:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            user_msg, bot_msg = pair
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))

    # Add the latest user input
    messages.append(HumanMessage(content=message))

    input_state = {"messages": messages}
    output = graph.invoke(input_state)

    # Extract the last message from LangGraph output
    final_message = output["messages"][-1]


    # Safe content extraction
    if isinstance(final_message, AIMessage):
        return final_message.content
    elif hasattr(final_message, "content") and isinstance(final_message.content, str):
        return final_message.content
    else:
        # Fallback: search backwards for last message with content
        for msg in reversed(output["messages"]):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                return msg.content

    return "Sorry, I couldn't generate a response."
