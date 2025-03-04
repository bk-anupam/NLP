import os
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf_processor import load_pdf, split_text, create_embeddings

def create_vectorstore(texts, embeddings, persist_directory="chroma_db"):
    """Creates a Chroma vectorstore from the texts and embeddings."""
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()  # Persist the vectorstore to disk
    return vectordb

def build_index(pdf_path, chunk_size=1000, chunk_overlap=200, persist_directory="chroma_db"):
    """Builds the index from the PDF document."""
    documents = load_pdf(pdf_path)
    texts = split_text(documents, chunk_size, chunk_overlap)
    embeddings = create_embeddings()
    vectordb = create_vectorstore(texts, embeddings, persist_directory)
    return vectordb

def load_existing_index(persist_directory="chroma_db"):
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