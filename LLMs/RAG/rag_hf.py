import gradio as gr
import numpy as np
import os
import torch
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Define functions for loading, splitting, and creating embeddings
def load_pdf(pdf_path):
    """Loads a PDF document from the given path."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    """Splits the documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings():
    """Creates embeddings using HuggingFaceEmbeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Or another suitable model
    return embeddings

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

# Define Gradio app functions
def upload_pdf(file):
    if not os.path.exists("chroma_db"):
        os.makedirs("chroma_db")
    pdf_path = file.name
    vectordb = build_index(pdf_path, persist_directory="chroma_db")
    return "PDF uploaded and indexed successfully."

def query_pdf(query):
    vectordb = load_existing_index(persist_directory="chroma_db")
    result = query_index(vectordb, query)
    return result

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# PDF Document Query App")
    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_button = gr.Button("Upload and Index PDF")
        upload_output = gr.Textbox()
        upload_button.click(upload_pdf, inputs=pdf_input, outputs=upload_output)
    with gr.Tab("Query PDF"):
        query_input = gr.Textbox(label="Enter your query")
        query_button = gr.Button("Query")
        query_output = gr.Textbox()
        query_button.click(query_pdf, inputs=query_input, outputs=query_output)

# Launch the Gradio app
demo.launch()