"""
Content‑retrieval service for the Multi‑Modal RAG system.
Now chat‑aware: keeps conversation history per session_id.
"""

import json, traceback, datetime
from base64 import b64decode
from typing import List, Tuple

import weaviate
import weaviate.classes.query as wq
from components.base_component import BaseComponent
from services.bedrock import MLLM
from app.prompt import user_query_prompt
from settings import settings

#  Helpers
def build_prompt(
    chat_history: str,
    text_context: List[str],
    image_context: List[str],
    question: str,
) -> list:
    """
    Compose the multimodal prompt for the LLM.

    chat_history –string containing previous turns
    text_context –list of paragraphs retrieved from Weaviate
    image_context –list of base‑64 PNG strings
    question –current user question
    """
    # prepend conversation memory
    prompt_text = f"Conversation so far:\n{chat_history}\n\n"

    #  add retrieved text context
    if text_context:
        prompt_text += "Document context:\n"
        prompt_text += "\n".join(text_context)

    #  add the template with the current question
    prompt = user_query_prompt.format(
        context_text=prompt_text,  # already included above
        user_question=question,
    )
    content = [{"type": "text", "text": prompt}]

    #  attach images
    for img in image_context:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img,
                },
            }
        )

    return content


#  Retriever
# noinspection PyPackageRequirements
class Retriever(BaseComponent):
    """
    Chat‑aware Retriever that:
      • pulls last N turns from MongoDB
      • appends them to the prompt
      • stores every new turn for future use
    """

    def __init__(self,app, session_id: str, history_limit: int = 5):
        super().__init__("Retriever")
        self.app = app  # reference to the FastAPI app instance
        self.session_id = session_id
        self.history_limit = history_limit

        # Bedrock LLM wrapper
        self.model = MLLM()
        self.doc_store = self.app.state.doc_store  # MongoDB client



    def run(self, question : str, queries: List[str]):
        """
        • queries == list from Query_decomposer
        • question == original user question (for history + prompt)
        """
        image_context, text_context, user_refs = [], [], []
        llm_response = {"status": 1, "answer": ""}

        try:
            client: weaviate.Client = self.app.state.vector_db.client
            collection = client.collections.get("DocumentCollection")

            # hybrid search for every decomposed query
            reference_docs = {}
            for q in queries:
                self.logger.info(f"Hybrid search for query: {q}")
                res = collection.query.hybrid(
                    query=q, limit=settings.search_limit, return_metadata=wq.MetadataQuery(score=True)
                )
                for obj in res.objects:
                    score = float(f"{obj.metadata.score:.3f}")
                    # thresholding the score to 0.7 to find the most relevant documents
                    if score >= settings.score_threshold:
                        # setdefault ensures unique UUIDs only once
                        reference_docs.setdefault(
                            obj.uuid,
                            {
                                "text": obj.properties["text"],
                                "score": f"{score:.3f}",
                            },
                        )
            #  ranking the  fetched context on the basis of the score
            sorted_reference_docs = dict(
                sorted(reference_docs.items(), key=lambda x: float(x[1]["score"]), reverse=True)
            )

            #  keep only the top 3 results
            reference_docs = dict(list(sorted_reference_docs.items())[:settings.ranking_limit])

            self.logger.info(f"Hybrid search results: {reference_docs}")
            # fetch metadata / raw content from MongoDB
            for uuid, ref in reference_docs.items():
                meta = self.doc_store.get_metadata(str(uuid))
                self.logger.info(f"Retrieved metadata for {uuid}: {meta}")
                if "image" in meta["metadata"]:
                    # it's an image
                    try:
                        b64decode(meta["metadata"]["image"])
                        image_context.append(meta["metadata"]["image"])
                    except Exception:
                        self.logger.error("Bad image b64", exc_info=True)
                    page = meta["metadata"]["metadata"]["page_number"]
                else:
                    # plain text
                    text_context.append(ref["text"])
                    page = meta["metadata"]["page_number"]

                user_refs.append(
                    {
                        "page_no": page,
                        "text": ref["text"],
                        "score": ref["score"],
                    }
                )

            # build prompt (includes chat history)
            chat_history = self.doc_store.get_chat_history(self.session_id, self.history_limit)
            prompt = build_prompt(chat_history, text_context, image_context, question)
            self.logger.info(f"prompt={prompt}")
            # hit the LLM
            raw = self.model.run(prompt)
            self.logger.info(f"raw response={raw}")
            llm_response = json.loads(raw)

            #  storing the chat history
            if llm_response["status"] == 1:
                self.doc_store.store_chat(question, llm_response["answer"],self.session_id)
            else:
                # irrelevant: ignore refs/ctx
                user_refs, image_context = [], []

        except Exception:
            self.logger.error("Retriever failure", exc_info=True)

        return llm_response["answer"], user_refs, image_context
