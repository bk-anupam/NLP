import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil # Import shutil for file operations
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Define functions for loading, splitting, and creating embeddings
def load_pdf(pdf_path):
    """Loads a PDF document from the given path."""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            print(f"Warning: No documents loaded from {pdf_path}. PDF might be empty or unreadable.")
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF from {pdf_path}: {e}")

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    """Splits the documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    if not texts:
        print("Warning: No text chunks generated. Document might be empty or splitting failed.")
    return texts

def create_embeddings():
    """Creates embeddings using HuggingFaceEmbeddings."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Or another suitable model
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to create HuggingFace embeddings: {e}. Check model name and internet connection.")

def create_vectorstore(texts, embeddings, persist_directory="chroma_db"):
    """Creates a Chroma vectorstore from the texts and embeddings."""
    try:
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
        return vectordb
    except Exception as e:
        raise RuntimeError(f"Failed to create Chroma vectorstore in {persist_directory}: {e}")

def build_index(pdf_path, chunk_size=1000, chunk_overlap=200, persist_directory="chroma_db"):
    """Builds the index from the PDF document."""
    print(f"Starting index build for: {pdf_path}")
    try:
        documents = load_pdf(pdf_path)
        texts = split_text(documents, chunk_size, chunk_overlap)
        embeddings = create_embeddings()
        vectordb = create_vectorstore(texts, embeddings, persist_directory)
        print(f"Index built successfully for: {pdf_path}")
        return vectordb
    except Exception as e:
        raise RuntimeError(f"Error building index for {pdf_path}: {e}")

def load_existing_index(persist_directory="chroma_db"):
    """Loads an existing Chroma vectorstore from disk."""
    try:
        embeddings = create_embeddings()  # Make sure you use the same embedding model used during indexing
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectordb
    except Exception as e:
        raise RuntimeError(f"Failed to load existing Chroma vectorstore from {persist_directory}: {e}")

def query_index(vectordb, query, chain_type="stuff", k=8, model_name="gemini-2.0-flash"):
    """Queries the vectorstore and returns the answer."""
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)  # Using chat generative model
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=vectordb.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True  # Optional: Return the source documents used for the answer
        )
        result = qa.invoke({"query": query})
        
        # Log the query and the retrieved context to the console
        print("-" * 50)
        print(f"Query: {result.get('query')}")
        print("\nRetrieved Context (Source Documents):")
        for i, doc in enumerate(result.get("source_documents", [])):
            print(f"\n--- Document {i+1} ---")
            print(f"Content: {doc.page_content}")            
        print("-" * 50)

        return result["result"]
    except Exception as e:
        raise RuntimeError(f"Failed to query index: {e}. Ensure Google API key is set and model is accessible.")

# Define Gradio app functions
def upload_pdf(file):
    if file is None:
        return "Error: No PDF file uploaded. Please select a file."

    upload_dir = "uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)

    pdf_filename = os.path.basename(file.name)
    permanent_pdf_path = os.path.join(upload_dir, pdf_filename)

    try:
        shutil.copy(file.name, permanent_pdf_path)
        print(f"Copied temporary file '{file.name}' to '{permanent_pdf_path}'")
    except Exception as e:
        return f"Error: Failed to copy uploaded PDF file. Details: {e}"

    chroma_db_dir = "chroma_db"
    try:
        build_index(permanent_pdf_path, persist_directory=chroma_db_dir)
        return f"PDF '{pdf_filename}' uploaded and indexed successfully."
    except Exception as e:
        return f"Error during PDF indexing: {e}"

def query_pdf(query):
    chroma_db_dir = "chroma_db"
    if not os.path.exists(chroma_db_dir) or not os.listdir(chroma_db_dir):
        return "Error: No indexed PDF found. Please upload and index a PDF first."
    try:
        vectordb = load_existing_index(persist_directory=chroma_db_dir)
        result = query_index(vectordb, query)
        return result
    except Exception as e:
        return f"Error during query: {e}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# PDF Document Query App")
    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_button = gr.Button("Upload and Index PDF")
        upload_output = gr.Textbox(label="Upload Status")
        upload_button.click(upload_pdf, inputs=pdf_input, outputs=upload_output)
    with gr.Tab("Query PDF"):
        query_input = gr.Textbox(label="Enter your query")
        query_button = gr.Button("Query")
        query_output = gr.Textbox(label="Query Result")
        query_button.click(query_pdf, inputs=query_input, outputs=query_output)

# Launch the Gradio app
demo.launch()


# As per the learner license document what precaution does the driver need to take while driving with learner license
# Specifically which rule of the CMV rules apply to a learning license holder while driving
# What are the expansion and reduction steps in openfe