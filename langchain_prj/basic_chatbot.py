from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    message: str

def greeting_node(state: AgentState) -> AgentState:
    """
    Simple node that generates a greeting message based on the user's input.
    """
    user_message = state['message']
    return {"message": f"Hello! You said: '{user_message}'. How can I assist you today?"}

graph = StateGraph(AgentState)
graph.add_node('greeting', greeting_node)
graph.set_entry_point('greeting')
graph.add_edge('greeting', END)

app = graph.compile()

result = app.invoke({"message": "Hi there!"})
print(result)

