import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=1024)

knowledge_base = os.path.join(os.path.dirname(__file__), "knowledge", "the-asset-january-2026.pdf")

if not os.path.exists(knowledge_base):
    raise FileNotFoundError(f"Knowledge base file not found: {knowledge_base}")

pdf_loader = PyPDFLoader(knowledge_base)
documents = pdf_loader.load()

# check if the pdf exists and is loaded correctly
try:
    print(f"Loaded {len(documents)} documents from the PDF.")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

#chunking the documents into smaller pieces for better retrieval

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages_split = text_splitter.split_documents(documents)
try:
    pc = Pinecone()
    index = pc.Index("hsbc-fund-knowledge")
    index_size = index.describe_index_stats()['total_vector_count']
    print("\nSize of vector data:"+f"{index_size}")
    vector_store = PineconeVectorStore(index=index, embedding = embeddings)
    if (index_size == 0):
         vector_store = PineconeVectorStore.from_documents(pages_split, embeddings, index_name="hsbc-fund-knowledge")
         print("Pinecone vector store created successfully.")
except Exception as e:
    print(f"Error creating Pinecone vector store: {e}")
    raise

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@tool
def retrieve_relevant_information(query: str) -> Sequence[str]:
    """Retrieves relevant information from the HSBC mutual fund document based on the query."""
    try:
        document = retriever.invoke(query)
        return [doc.page_content for doc in document]
    except Exception as e:
        print(f"Error retrieving information: {e}")
        return []

tools = [retrieve_relevant_information]
llm = llm.bind_tools(tools) 

    
class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state:AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
system_prompt = """
You are an AI assistant to answer questions about HSBC mutual funds. 
You have access to a tool that can retrieve relevant information from the HSBC mutual fund document. When you receive a question, use the tool to retrieve relevant information and then answer the question based on that information. If you have already retrieved information and can answer the question, then do not use the tool again. Always try to answer the question as best as you can with the information you have.
Please cite the specific parts of the document you used to answer the question in your response.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

#LLM agent
def model_call(state:AgentState) -> AgentState:
    messages = state['messages']
    messages = [SystemMessage(content=system_prompt)]+messages
    response = llm.invoke(messages)
    return {"messages": [response]}

#retirever agent
def retriever_agent_call(state:AgentState) -> AgentState:
    """Execute tools calls from the llms response and add the results to the messages."""
    tool_calls = state['messages'][-1].tool_calls
    results =[]
    for t in tool_calls:
        print (f"Executing tool: {t['name']}")
        if not t['name'] in tools_dict:
            print(f"Tool {t['name']} not found in tools_dict.")
            result = "Incorrect tool name"
        else:
            result = tools_dict[t['name']].invoke(t['args'].get("query", ''))
            print(f"Result from tool {t['name']}: {result}")
        results.append(ToolMessage(tool_call_id=t['id'], tool_name=t['name'], content=result))
    
    print("tool execution completed")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("model_call", model_call)
graph.add_node("retriever_agent_call", retriever_agent_call)

graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue": "retriever_agent_call",
        "end": END
    }
)    
graph.add_edge("retriever_agent_call", "model_call")
graph.set_entry_point("model_call")
app = graph.compile()

def running_agent():
    while True:
        user_input = input("What is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the agent.")
            return
        messages = [HumanMessage(content=user_input)]
        result = app.invoke({"messages": messages})
        print("\nAgent response:")
        for message in result['messages']:
            message.pretty_print()
        
if __name__ == "__main__":
    running_agent()