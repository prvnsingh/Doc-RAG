from components.base_component import BaseComponent
from .bedrock import MLLM
from base64 import b64decode
from typing import List, Dict, Any
from app.prompt import user_query_prompt
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
import traceback
from langchain.storage import InMemoryStore


class RetrieverV2(BaseComponent):
    def __init__(self,retriever):
        super().__init__('RetrieverV2')
        persist_directory = "resources/chroma_langchain_db"
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        self.vector_db = Chroma(collection_name="MRAG_Mech_book",persist_directory=persist_directory, embedding_function=embeddings)
        self.retriever = retriever

        self.model = MLLM()

    def parse_docs(self, docs):
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            # content = doc.page_content
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}

    def build_prompt(self, context, question):
        """Build a prompt with context and question"""
    
        context_text = ""
        if len(context["texts"]) > 0:
            for text_element in context["texts"]:
                context_text += text_element.page_content

        content = [{"type": "text", "text": user_query_prompt.format(context_text = context_text,user_question=question)}]


        if len(context["images"]) > 0:
            for image in context["images"]:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                            "data": image}})

        return content

    def run(self, question: str) -> str:
        """Run the retriever pipeline"""
        response = ''

        try:
            # retrievering the context based on the question 
            # docs = self.vector_db.similarity_search(question)
            docs = self.retriever.invoke(question)
            # self.retriever.doc
            for doc in docs:
                self.logger.info(str(doc) + "\n\n" + "-" * 80)
            # Parse documents into images and text
            parsed_docs = self.parse_docs(docs)
            self.logger.info(f'{parsed_docs=}')
            # Build the prompt with context
            prompt = self.build_prompt(parsed_docs, question)
            self.logger.info(f'{prompt=}')
            
            # Get response from the model
            response = self.model.run(prompt)
        except Exception as e:
            traceback.print_exc()
        return response