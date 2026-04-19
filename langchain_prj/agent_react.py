import random
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

def multiply(a: int, b: int) -> int:    
    """Multiply two integers."""
    return a * b

def subtract(a: int, b: int) -> int:
    """Subtract two integers."""
    return a - b


tools = [add, multiply, subtract]

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.7)
model = model.bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content =
                                    "You are my AI assistant. Please respond to the user's message and use the add tool if needed."
                                 )
    # Simulate processing messages and generating a response
    response = model.invoke([system_prompt] + list(state['messages']))
    return {"messages": [response]}

def should_continue(state:AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.add_node("our_agent", model_call)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [HumanMessage(content="Add 20+32 and then multiply the result with 6")]}
print_stream(app.stream(inputs, stream_mode="values"))
