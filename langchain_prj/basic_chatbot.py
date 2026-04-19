from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    number1: int
    number2: int
    operation: str
    finalnumber: int

def adder(state: AgentState) -> AgentState:
    """
    Thiscnode qdds two numbers
    """
    state['finalnumber'] = state['number1'] + state['number2']
    return state

def subtractor(state: AgentState) -> AgentState:
    """
    This node subtracts two numbers
    """
    state['finalnumber'] = state['number1'] - state['number2']
    return state


def decide_next_noode(state: AgentState) -> str:
    """
    This node decides the next node based on the operation
    """
    if state['operation'] == 'add':
        return 'addition_operation'
    else:
        return 'subtraction_operation'
    


graph = StateGraph(AgentState)
graph.add_node('adder', adder)
graph.add_node('subtractor', subtractor)
graph.add_node ('router', lambda state: state)     
graph.add_edge(START, "router")
graph.add_conditional_edges("router", 
                           decide_next_noode, 
                           {
                                 'addition_operation': 'adder',
                                 'subtraction_operation': 'subtractor'
                            })
graph.add_edge('adder', END)
graph.add_edge('subtractor', END)   
app = graph.compile()

result = app.invoke(AgentState(number1=10, number2=5, operation='add', finalnumber=0)   )
print(result)

