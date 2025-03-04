import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

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