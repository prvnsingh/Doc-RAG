"""
Vector embedding and storage service.
This module provides functionality to create and store vector embeddings for text, tables, and images
using AWS Bedrock embeddings and Chroma vector store.
"""

import uuid
from typing import List
from langchain_core.documents import Document
from components.base_component import BaseComponent
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from settings import settings

class Vectorizer(BaseComponent):
    """
    Vector embedding and storage manager for multi-modal content.
    
    This class handles the creation and storage of vector embeddings for different types of content:
    - Text passages
    - Tables
    - Images
    
    It uses AWS Bedrock for embeddings and Chroma for vector storage, with an in-memory store
    for the original content.
    
    Attributes:
        embedding_function: AWS Bedrock embedding model
        vector_store: Chroma vector store instance
        id_key (str): Key used for document identification
    """

    def __init__(self):
        """Initialize the vectorizer with embedding model and storage components."""
        super().__init__('Vectorizer')

        # Initialize AWS Bedrock embedding model
        self.embedding_function = BedrockEmbeddings(model_id=settings.embedding_model)
        
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name="MRAG_Mech_book",
            embedding_function=self.embedding_function,
            persist_directory=settings.persist_directory,  # Where to save data locally
        )

        self.id_key = "doc_id"
        self.original_doc = "original_doc"


    def run(
        self,
        texts: List[str],
        tables: List[str],
        images: List[str],
        text_summaries: List[str],
        table_summaries: List[str],
        image_summaries: List[str]
    ):
        """
        Process and store vector embeddings for all content types.
        
        This method:
        1. Creates unique IDs for each content item
        2. Stores summaries in the vector store
        3. Stores original content in the doc store
        4. Links them using the unique IDs
        5. Metadata stores the original document. 
        A dedicated persistant store (docstore) should be used for production
        
        Args:
            texts (List[str]): Original text content
            tables (List[str]): Original table content
            images (List[str]): Original image content
            text_summaries (List[str]): Summaries of text content
            table_summaries (List[str]): Summaries of table content
            image_summaries (List[str]): Summaries of image content
            
        """
        # Process text content and summaries
        if text_summaries:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            summary_texts = [
                Document(page_content=summary, metadata={self.id_key: doc_ids[i],self.original_doc: str(texts[i])}) 
                for i, summary in enumerate(text_summaries)
            ]
            self.vector_store.add_documents(summary_texts)

        # Process table content and summaries
        if table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i],self.original_doc: str(tables[i])}) 
                for i, summary in enumerate(table_summaries)
            ]
            self.vector_store.add_documents(summary_tables)

        # Process image content and summaries
        if image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in images]
            summary_img = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i],self.original_doc: str(images[i])}) 
                for i, summary in enumerate(image_summaries)
            ]
            self.vector_store.add_documents(summary_img)

