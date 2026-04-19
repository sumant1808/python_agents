import random
from typing import Annotated
from urllib import response
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages:list [HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.7)

def process_messages(state:AgentState) -> AgentState:
    # Simulate processing messages and generating a response
    response = llm.invoke(state['messages'])
    content = response.content
    text = content if isinstance(content, str) else " ".join(b['text'] for b in content if b.get('type') == 'text')
    print(f"\nAI: {text}\n")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process_messages)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")  

    