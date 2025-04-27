import uuid
from typing import List
import numpy as np
from langchain_core.documents import Document
from components.base_component import BaseComponent
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

class VectorizerV2(BaseComponent):

    def __init__(self):
        super().__init__('VectorizerV2')

        # 1) Embedding function
        self.embedding_function = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        

        self.vector_store = Chroma(
            collection_name="MRAG_Mech_book",
            embedding_function=self.embedding_function,
            persist_directory="resources/chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )

        # 3) Parent docstore for the original texts/tables/images
        self.id_key = "doc_id"
        self.store = InMemoryStore()
        # The retriever (empty to start)
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=self.store,
            id_key=self.id_key,
)


    def run(
        self,
        texts: List[str],
        tables: List[str],
        images: List[str],
        text_summaries: List[str],
        table_summaries: List[str],
        image_summaries: List[str]
    ):
        if text_summaries:
            # Add texts
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            summary_texts = [
                Document(page_content=summary, metadata={self.id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(doc_ids, texts)))

        if table_summaries:
            # Add tables
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, tables)))

        if image_summaries:
            # Add image summaries
            img_ids = [str(uuid.uuid4()) for _ in images]
            summary_img = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_img)
            self.retriever.docstore.mset(list(zip(img_ids, images)))

        return self.retriever