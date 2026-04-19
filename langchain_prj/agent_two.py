import random
from typing import Annotated, Union
from urllib import response
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: list[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.7)

def process_messages(state:AgentState) -> AgentState:
    # Simulate processing messages and generating a response
    response = llm.invoke(state['messages'])
    state['messages'].append(response)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process_messages)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})   
    print("AI Response:", result['messages'][-1].content)
    conversation_history = result['messages']  # Update conversation history with AI response
    user_input = input("Enter your message: ")  

with open("conversation_history.txt", "w") as f:
    f.write("Conversation History:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("end of conversation\n")

print("Conversation history saved to conversation_history.txt")