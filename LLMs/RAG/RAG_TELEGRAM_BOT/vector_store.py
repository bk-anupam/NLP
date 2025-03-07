import os
import sys
from logger import logger
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from pdf_processor import load_pdf, split_text, create_embeddings
from config import Config

def create_vectorstore(texts, embeddings, persist_directory=Config.VECTOR_STORE_PATH):
    """
    Creates a Chroma vectorstore from the texts and embeddings, or loads an existing one and updates it with new texts.
    Args:
        texts (list): A list of text documents to be added to the vector store.
        embeddings (callable): A function or model that generates embeddings for the texts.
        persist_directory (str, optional): The directory path where the vector store will be persisted. 
        Defaults to Config.VECTOR_STORE_PATH.
    Returns:
        Chroma: The updated or newly created Chroma vector store.
    """    
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # Load existing vector store
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # vectordb.load()  # Load the vectorstore from disk
        # Add new documents to the existing vector store
        vectordb.add_documents(documents=texts)
    else:
        # Create a new vector store
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    
    #vectordb.persist()  # Persist the updated vectorstore to disk
    logger.info(f"Vector store updated with {len(texts)} documents")    
    return vectordb


def build_index(pdf_path, chunk_size=1000, chunk_overlap=200, persist_directory=Config.VECTOR_STORE_PATH):
    """Builds the index from the PDF document."""
    documents = load_pdf(pdf_path)
    texts = split_text(documents, chunk_size, chunk_overlap)
    embeddings = create_embeddings()
    vectordb = create_vectorstore(texts, embeddings, persist_directory)
    return vectordb


def load_existing_index(persist_directory=Config.VECTOR_STORE_PATH):
    """Loads an existing Chroma vectorstore from disk."""
    embeddings = create_embeddings()  # Make sure you use the same embedding model used during indexing
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb


def query_index(vectordb, query, chain_type="stuff", k=4, model_name="gemini-2.0-flash"):
    """Queries the vectorstore and returns the answer."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)  # Using chat generative model
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=vectordb.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True  # Optional: Return the source documents used for the answer
    )
    result = qa({"query": query})
    return result["result"]


def query_llm(query, model_name="gemini-2.0-flash"):
    """Queries the LLM directly and returns the answer."""
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)        
        # Format the query properly as a message
        messages = [HumanMessage(content=query)]        
        # Get the response
        response = llm.invoke(messages)        
        # Extract the content from the response
        return response.content
    except Exception as e:
        print(f"Error querying LLM: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"
    

# add code to execute vector_store.py as a script
if __name__ == "__main__":    
    # if len(sys.argv) != 2:
    #     print("Usage: python vector_store.py <pdf_path>")
    #     sys.exit(1)
    # pdf_path = sys.argv[1]
    pdf_path = "/home/bk_anupam/code/ML/NLP/LLMs/RAG/data"
    config = Config()
    print(f"Building index for {pdf_path}")
    pdf_files = [os.path.join(pdf_path, f) for f in os.listdir(pdf_path) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        print(f"Building index for {pdf_file}")
        vectordb = build_index(pdf_file)
    print("Indexing complete.")


    # # recursively loop through  a directory till you find all the pdf documents and build index for each
    # if os.path.isdir(pdf_path):
    #     for root, dirs, files in os.walk(pdf_path):
    #         for file in files:
    #             if file.endswith(".pdf"):
    #                 pdf_file = os.path.join(root, file)
    #                 print(f"Building index for {pdf_file}")
    #                 vectordb = build_index(pdf_file)
    # else:
    #     print(f"Building index for {pdf_path}")
    #     vectordb = build_index(pdf_path)
    
    # # vectordb = build_index(pdf_path)
    # print("Indexing complete.")