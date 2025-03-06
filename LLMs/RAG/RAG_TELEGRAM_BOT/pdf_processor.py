import os
from logger import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf(pdf_path):
    """Loads a PDF document from the given path."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from {pdf_path}")
    return documents

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    """Splits the documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(texts)} chunks")
    return texts

def create_embeddings():
    """Creates embeddings using HuggingFaceEmbeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Or another suitable model
    return embeddings