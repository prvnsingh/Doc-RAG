import uuid
from typing import List
import faiss
import numpy as np
from langchain_core.documents import Document
from components.base_component import BaseComponent
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

class Vectorizer(BaseComponent):

    def __init__(self):
        super().__init__('Vectorizer')

        # 1) Embedding function
        self.embedding_function = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        
        # 2) Manually construct an empty FAISS index
        dimension = len(self.embedding_function.embed_query("hello world"))
        index = faiss.IndexFlatL2(dimension)

        self.vector_store = FAISS(
            embedding_function=self.embedding_function,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # 3) Parent docstore for the original texts/tables/images
        self.id_key = "doc_id"


    def run(
        self,
        texts: List[str],
        tables: List[str],
        images: List[str],
        text_summaries: List[str],
        table_summaries: List[str],
        image_summaries: List[str]
    ):
        # — TEXTS —
        if texts and text_summaries:
            text_ids = [str(uuid.uuid4()) for _ in texts]
            text_docs = [
                Document(page_content=summary, metadata={self.id_key: tid})
                for summary, tid in zip(text_summaries, text_ids)
            ]
            # Safely add documents (only if there are any)
            if text_docs:
                self._safe_add_documents(text_docs)
                # Save full parent texts
                for tid, text in zip(text_ids, texts):
                    self.vector_store.docstore.add({tid: text})

        # — TABLES —
        if tables and table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            table_docs = [
                Document(page_content=summary, metadata={self.id_key: tid})
                for summary, tid in zip(table_summaries, table_ids)
            ]
            # Safely add documents (only if there are any)
            if table_docs:
                self._safe_add_documents(table_docs)
                # Save full parent tables
                for tid, table in zip(table_ids, tables):
                    self.vector_store.docstore.add({tid: table})

        # — IMAGES —
        if images and image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in images]
            img_docs = [
                Document(page_content=summary, metadata={self.id_key: iid})
                for summary, iid in zip(image_summaries, img_ids)
            ]
            # Safely add documents (only if there are any)
            if img_docs:
                self._safe_add_documents(img_docs)
                # Save full parent images
                for iid, image in zip(img_ids, images):
                    self.vector_store.docstore.add({iid: image})

        # 5) Persist FAISS index
        self.vector_store.save_local("resources/faiss_index")
        
    def _safe_add_documents(self, docs):
        """Safely add documents to FAISS by directly handling embeddings."""
        if not docs:
            return
            
        # Get embeddings directly
        embeddings = self.embedding_function.embed_documents([doc.page_content for doc in docs])
        
        # Ensure embeddings are in the right format for FAISS (2D array)
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            # Reshape if needed - FAISS expects a 2D array (n_vectors x dimension)
            if len(embeddings_array.shape) == 1:
                # If we have a single vector, reshape to a 2D array with one row
                dimension = embeddings_array.shape[0]
                embeddings_array = embeddings_array.reshape(1, dimension)
                
            # Add embeddings to the index
            self.vector_store.index.add(embeddings_array)
            
            # Add document mapping
            for i, doc in enumerate(docs):
                idx = self.vector_store.index.ntotal - len(docs) + i
                self.vector_store.index_to_docstore_id[idx] = doc.metadata[self.id_key]
