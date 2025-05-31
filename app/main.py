"""
Main FastAPI application module for the Multi-Modal RAG (Retrieval-Augmented Generation) system.
This module provides endpoints for file processing, question answering, and query decomposition.
"""

from settings import settings
import uvicorn
from fastapi import FastAPI, UploadFile, File
from services.extractor import Extractor
from services.vectorDB import VectorDB
from services.summarizer import Summarizer
from services.retriever import Retriever
from services.query_dcomposer import Query_decomposer
import time

# Initialize FastAPI application with configuration from settings
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
    # lifespan=lifespan,
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
def embedding_file(file: UploadFile = File(...)):
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
    # print(f' embedding file {file.filename}')
    file_content = file.file
    extractor = Extractor()
    extractor.run(file_content)
    summarizer = Summarizer(extractor.texts, extractor.tables, extractor.images_b64)
    data = summarizer.run()
    vectorizer = VectorDB()
    vectorizer.run(data)
    et = time.time()
    print(f'time taken {et - st}')
    return {"status": "success"}


@app.get("/ask_question")
def query_from_user(question: str):
    """
    Process a user question and return an answer with relevant context.
    
    Args:
        question (str): The user's question
        
    Returns:
        dict: Contains the answer and relevant context (text and images)
    """
    decomposer = Query_decomposer()
    queries = decomposer.run(question)
    retriever = Retriever()
    llm_response, fetch_context_text, fetch_context_image = retriever.run(queries)
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


if __name__ == '__main__':
    uvicorn.run(app, port=settings.port, host=settings.host)
