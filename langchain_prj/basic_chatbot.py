from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

from langchain_gemini import Gemini

class State(TypedDict):
    messages: Annotated [list[str], add_messages]

graph_builder = StateGraph (State)