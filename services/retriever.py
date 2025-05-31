"""
Content retrieval service for the Multi-Modal RAG system.
This module provides functionality to retrieve relevant content (text and images) based on user queries
and generate responses using a language model.
"""
import json
import weaviate
from settings import settings
from components.base_component import BaseComponent
from .bedrock import MLLM
from base64 import b64decode
import weaviate.classes.query as wq
from app.prompt import user_query_prompt
import traceback
from app.config import config
from services.document_store import DocumentStore


class Retriever(BaseComponent):

    def __init__(self):
        """
        Initialize the retriever with necessary components.
        """
        super().__init__('Retriever')
        self.headers = {
            "X-AWS-Access-Key": config.AWS_ACCESS_KEY_ID,
            "X-AWS-Secret-Key": config.AWS_SECRET_ACCESS_KEY,
        }
        self.model = MLLM()

    def parse_docs(self, docs, metadatas):
        """
        Separate base64-encoded images from text content.

        Args:
            docs: List of retrieved documents

        Returns:
            dict: Dictionary containing separated images and texts

        Parameters
        ----------

        metadatas
        """
        b64 = []
        # text = []
        for metadata in metadatas:
            if 'image' in metadata['metadata'].keys():
                try:
                    b64decode(metadata['metadata']['image'])
                    b64.append(metadata['metadata']['image'])
                except Exception:
                    print("got error")
        return {"images": b64, "texts": docs}

    def build_prompt(self, text_context, image_context, question):
        """
        Build a prompt combining context and question for the language model.
        
        Args:
            context (dict): Dictionary containing text and image context
            question (str): User's question
            
        Returns:
            tuple: (prompt content, text context for user, image context for user)

        Parameters
        ----------
        text_context
        image_context
        """
        context_text = ""


        # Process text context
        if len(text_context) > 0:
            for text_element in text_context:
                # Handle both Document and CompositeElement objects
                if hasattr(text_element, 'page_content'):
                    content = text_element.page_content
                else:
                    content = str(text_element)
                context_text += content

        # Build base prompt with text context
        content = [{
            "type": "text",
            "text": user_query_prompt.format(
                context_text=context_text,
                user_question=question
            )
        }]

        # Add image context if available
        if len(image_context) > 0:
            for image in image_context:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image
                    }
                })

        return content

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
        llm_response = {'answer':""}
        reference_docs = {}
        image_context = []
        text_context = []
        user_references = []
        with weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051, headers=self.headers) as client:
            collection = client.collections.get("DocumentCollection")
            try:
                # Retrieve context for each question
                for question in questions:
                    response = collection.query.hybrid(
                        query=question,  # The model provider integration will automatically vectorize the query
                        limit=2,
                        return_metadata=wq.MetadataQuery(score=True)
                    )
                    for obj in response.objects:
                        if obj.uuid not in reference_docs.keys():
                            reference_docs[str(obj.uuid)] = {"text": obj.properties["text"],
                                                             "score": f"{obj.metadata.score:.3f}"}

                with DocumentStore(config.MONGO_URI) as doc_store:
                    for uuid, reference in reference_docs.items():
                        metadata_doc = doc_store.get_metadata(uuid)
                        ref_page_no = 0
                        if 'image' in metadata_doc['metadata'].keys():
                            try:
                                b64decode(metadata_doc['metadata']['image'])
                                image_context.append(metadata_doc['metadata']['image'])
                                ref_page_no = metadata_doc['metadata']['metadata']['page_number']
                            except Exception as e:
                                self.logger.info(traceback.print_exc())
                        else:
                            text_context.append(reference['text'])
                            ref_page_no = metadata_doc['metadata']['page_number']

                        user_reference = {"page_no": ref_page_no, "text": reference['text'],"score": reference['score']}
                        user_references.append(user_reference)

                self.logger.info(f"user reference ====== {user_references}")

                # Build prompt and get response
                prompt = self.build_prompt(text_context, image_context, question)
                self.logger.info(f'{prompt=}')

                response = self.model.run(prompt)
                self.logger.info(f'{response}')
                llm_response = json.loads(response)
                # if the question is irrelevant or off-topic
                if llm_response['status'] == 0:
                    user_references = []
                    image_context = []

            except Exception as e:
                self.logger.info(traceback.print_exc())

            return llm_response['answer'], user_references, image_context
