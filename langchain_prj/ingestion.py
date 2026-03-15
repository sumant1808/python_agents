import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()  # Load environment variables from .env file


def main():
    print("ingestion")
    loader = PyPDFLoader("/Users/sumantkumar/Documents/Building-Societies-Report-2025.pdf")
    document = loader.load()
    print("document loaded")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"document split into {len(texts)} chunks")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    llm = ChatOllama(model="gemma3:270m")
    query = "Provide a summary of the Building Societies Report 2025"
    chain = PromptTemplate.from_template(template=query) | llm
    response = chain.invoke(input={})
    print("LLM response:"+response.content)


    vectorstore = PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ.get("INDEX_NAME"),
    )

    retrieval_qa_chat = hub.pull("langchain/chat-retrieval-qa")
    combined_chain = create_stuff_documents_chain(llm,retrieval_qa_chat)
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(),
        combined_chain,
    )
    result = retrieval_chain.invoke(input={"input": query})
    print (result)    
    

if __name__ == "__main__":
    main()