"""
Main FastAPI application module for the Multi-Modal RAG (Retrieval-Augmented Generation) system.
This module provides endpoints for file processing, question answering, and query decomposition.
"""
from app.config import config
from services.document_store import DocumentStore
from settings import settings
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File,Request
from services.extractor import Extractor
from services.vectorDB import VectorDB
from services.summarizer import Summarizer
from services.retriever import Retriever
from services.query_dcomposer import Query_decomposer
import time
import warnings
import re
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle the lifespan of the FastAPI application.
    This is where you can initialize resources or start background tasks.
    """
    try:
        print("Application is starting up...")
        # Initialize the vector database connection
        vector_db = VectorDB()
        app.state.vector_db = vector_db

        # initialize document store
        document_store = DocumentStore(config.MONGO_URI)
        app.state.doc_store = document_store

        yield
    finally:
        print("Application is shutting down...")
        if hasattr(app.state, "vector_db"):
            app.state.vector_db.client.close()

        if hasattr(app.state, "doc_store"):
            app.state.doc_store.client.close()



# Initialize FastAPI application with configuration from settings
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict: A dictionary containing the status of the API
    """
    return {"status": "healthy"}


global vector_retriever


@app.post("/upload_file_for_embedding")
def embedding_file(request: Request, file: UploadFile = File(...)):
    session_id = request.headers.get("session-id", "unknown")
    """
    Process and embed a file for later retrieval.
    This endpoint handles the complete pipeline of:
    1. Extracting content from the file
    2. Summarizing the content
    3. Creating vector embeddings for retrieval
    
    Args:
        file (UploadFile): The file to be processed
        
    Returns:
        dict: Status of the embedding process
    """
    st = time.time()
    print(f"Session ID: {session_id}")
    # print(f' embedding file {file.filename}')
    file_content = file.file
    extractor = Extractor()
    extractor.run(file_content)

    summarizer = Summarizer(extractor.texts, extractor.tables, extractor.images_b64)
    data = summarizer.run()

    app.state.vector_db.run(data,app)
    et = time.time()
    print(f'time taken {et - st}')
    return {"status": "success"}


@app.get("/ask_question")
def query_from_user(request: Request, question: str):
    """
    Process a user question and return an answer with relevant context.
    
    Args:
        request:
        question (str): The user's question
        
    Returns:
        dict: Contains the answer and relevant context (text and images)
    """
    session_id = request.headers.get("session-id", "unknown")
    print(f"Session ID: {session_id}")

    decomposer = Query_decomposer()
    queries = decomposer.run(question)
    if len(queries) == 1:
        return {
            "answer": queries[0],
            "context_texts": [],
            "context_images": []
        }
    retriever = Retriever(app,session_id=session_id)  # Update Retriever to accept session_id
    llm_response, fetch_context_text, fetch_context_image = retriever.run(question, queries)
    return {
        "answer": llm_response,
        "context_texts": fetch_context_text,
        "context_images": fetch_context_image
    }


@app.get("/query_decompose")
def query_from_user(question: str):
    """
    Decompose a complex question into simpler sub-queries.
    
    Args:
        question (str): The complex question to decompose
        
    Returns:
        list: Decomposed sub-queries
    """
    decomposer = Query_decomposer()
    response = decomposer.run(question)
    return response


@app.post("/extracting_from_file")
def embedding_file(file: UploadFile = File(...)):
    """
    Extract content from a file without creating embeddings.
    
    Args:
        file (UploadFile): The file to extract content from
        
    Returns:
        dict: Status of the extraction process
    """
    # print(f' embedding file {file.filename}')
    file_content = file.file
    extractor = Extractor()
    extractor.run(file_content)

# Suppress specific Pydantic v2 deprecation warnings from libraries
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"(pydantic|nemoguardrails|langchain_community)"
)

if __name__ == '__main__':
    uvicorn.run(app, port=settings.port, host=settings.host)
