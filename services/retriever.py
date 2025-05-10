"""
Content retrieval service for the Multi-Modal RAG system.
This module provides functionality to retrieve relevant content (text and images) based on user queries
and generate responses using a language model.
"""
from settings import settings
from components.base_component import BaseComponent
from .bedrock import MLLM
from base64 import b64decode
from typing import List, Dict, Any
from app.prompt import user_query_prompt
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
import traceback


class Retriever(BaseComponent):
    """
    Content retriever that fetches relevant information based on user queries.
    
    This class handles:
    1. Retrieving relevant content from the vector store
    2. Processing and deduplicating retrieved documents
    3. Building prompts with context for the language model
    4. Generating responses using the language model
    
    Attributes:
        vector_db: Chroma vector database instance
        model: Language model instance for generating responses
        retriever: Multi-vector retriever for fetching content
    """

    def __init__(self):
        """
        Initialize the retriever with necessary components.
        """
        super().__init__('Retriever')
        embeddings = BedrockEmbeddings(model_id=settings.embedding_model)
        self.vector_db = Chroma(
            collection_name="MRAG_Mech_book",
            persist_directory=settings.persist_directory,
            embedding_function=embeddings
        )
        self.model = MLLM()
        self.retriever = self.vector_db.as_retriever()

    def parse_docs(self, docs):
        """
        Separate base64-encoded images from text content.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            dict: Dictionary containing separated images and texts
        """
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}

    def build_prompt(self, context, question):
        """
        Build a prompt combining context and question for the language model.
        
        Args:
            context (dict): Dictionary containing text and image context
            question (str): User's question
            
        Returns:
            tuple: (prompt content, text context for user, image context for user)
        """
        context_text = ""
        context_text_for_user = []
        context_image_for_user = []
        
        # Process text context
        if len(context["texts"]) > 0:
            for text_element in context["texts"]:
                # Handle both Document and CompositeElement objects
                if hasattr(text_element, 'page_content'):
                    content = text_element.page_content
                else:
                    content = str(text_element)
                context_text += content
                context_text_for_user.append(content)

        # Build base prompt with text context
        content = [{
            "type": "text",
            "text": user_query_prompt.format(
                context_text=context_text,
                user_question=question
            )
        }]

        # Add image context if available
        if len(context["images"]) > 0:
            for image in context["images"]:
                context_image_for_user.append(image)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image
                    }
                })

        return content, context_text_for_user, context_image_for_user

    def run(self, questions: list) -> str:
        """
        Execute the retrieval and response generation pipeline.
        
        This method:
        1. Retrieves relevant documents for each question
        2. Deduplicates the retrieved documents
        3. Separates text and image content
        4. Builds a prompt with the context
        5. Generates a response using the language model
        
        Args:
            questions (list): List of questions to process
            
        Returns:
            tuple: (generated response, retrieved text context, retrieved image context)
        """
        response = ''
        fetched_context_text = []
        fetched_context_image = []
        docs = []
        unique_docs = []
        seen_doc = set()
        
        try:
            # Retrieve context for each question
            for question in questions:
                docs.extend(self.retriever.invoke(question))
            
            # Deduplicate and process retrieved documents
            for doc in docs:
                self.logger.info(str(type(doc)))
                self.logger.info(doc)
                if doc.metadata['doc_id'] not in seen_doc:
                    self.logger.info(doc.metadata['doc_id'])
                    self.logger.info(doc.metadata['original_doc'])
                    seen_doc.add(doc.metadata['doc_id'])
                    unique_docs.append(doc.metadata['original_doc'])

            
            self.logger.info(f'number of docs fetched in total = {len(docs)}========== number of unique docs = {len(unique_docs)}')
            
 
            parsed_docs = self.parse_docs(unique_docs)
            
            # Build prompt and get response
            prompt, fetched_context_text, fetched_context_image = self.build_prompt(parsed_docs, question)
            self.logger.info(f'{prompt=}')
            
            response = self.model.run(prompt)
        except Exception as e:
            traceback.print_exc()
            
        return response, fetched_context_text, fetched_context_image