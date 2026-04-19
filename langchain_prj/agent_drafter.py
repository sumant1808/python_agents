import random
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_content(content:str) -> str:
    """Update the content of the message."""
    global document_content
    document_content = content
    return f"Content updated! Current content: {document_content}"

@tool
def save(filename:str) -> str:
    """Save the content to a file.
    
    Args:        filename: The name of the file to save the content to.
    
    """
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        print(f"Content saved to {filename}!")
        return f"Content saved to {filename}!"
    except Exception as e:
        print(f"Error saving content to {filename}: {e}")
        return f"Error saving content to {filename}: {e}"

tools = [update_content, save]

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.7)
model = model.bind_tools(tools)

def our_agent(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content =
                                    "You are a content drafter. " \
                                    "Please help the user to update por modify content" \
                                    "use the save tool to save the content"
                                    f"The current document content is {document_content}"
                                 )
    if not state['messages']:
        user_inmput = HumanMessage(content="Please help me to draft a content for my blog post about AI.")
    else:
        user_inmput = HumanMessage(content="What would you like me to do with the document?")
    all_messages = [system_prompt] + list(state['messages'])+ [user_inmput]
    # Simulate processing messages and generating a response
    response = model.invoke(all_messages)
    print   (f"Agent response: {response.content}")
    if hasattr  (response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool called: {tool_call['name']} with arguments {tool_call['args']}")   
    return {"messages": list(state['messages']) + [user_inmput,response]}

def should_continue(state:AgentState):
    messages = state['messages']
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        # check if this is a tool message and if the tool name is save
        if isinstance(message, ToolMessage) and \
           "saved" in message.content.lower() and \
           "document" in message.content.lower():
            return "end"
        
    return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", our_agent)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("our_agent")
graph.add_edge("our_agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "our_agent",
        "end": END
    }
)
app= graph.compile()

def print_messages(messages):
    """Utility function to print messages in a readable format."""
    if not messages:
        return
    for message in messages[-3:]:  # Print only the last 3 messages for brevity 
        if isinstance(message, ToolMessage):
            print(f"Tool: {message.content}")
        
def run_document_drafter():
    state = {"messages": []}
    
    for step in app.stream  (state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n drafter finished execution.")

if __name__ == "__main__":
    run_document_drafter()