from components.base_component import BaseComponent
from .vectorizer import Vectorizer
from .bedrock import MLLM
from base64 import b64decode
from typing import List, Dict, Any
from app.prompt import user_query_prompt
from langchain.vectorstores import FAISS
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_aws import BedrockEmbeddings


class Retriever(BaseComponent):
    def __init__(self):
        super().__init__('Retriever')
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        self.vector_db = FAISS.load_local(
            "resources/faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_db,
            id_key="doc_id",
        )
        self.model = MLLM()

    def parse_docs(self, docs: List[str]) -> Dict[str, List[str]]:
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}

    def build_prompt(self, context: Dict[str, List[str]], question: str) -> List[Dict[str, Any]]:
        """Build a prompt with context and question"""
    
        context_text = ""
        if len(context["texts"]) > 0:
            for text_element in context["texts"]:
                context_text += text_element

        content = [{"type": "text", "text": user_query_prompt.format(context_text = context_text,question=question)}]


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
            docs = self.retriever.invoke(question)
            self.logger.info(f'{docs=}')
            # Parse documents into images and text
            parsed_docs = self.parse_docs(docs)
            self.logger.info(f'{parsed_docs=}')
            # Build the prompt with context
            prompt = self.build_prompt(parsed_docs, question)
            self.logger.info(f'{prompt=}')
            
            # Get response from the model
            response = self.model.run(prompt)
        except Exception as e:
            print(e)
            print(e.with_traceback)
        return response