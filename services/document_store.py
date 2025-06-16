import datetime

from pymongo import MongoClient
from typing import Any, Dict, Optional


class DocumentStore:
    def __init__(self, mongo_uri, db_name="doc_gpt"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.meta_col = self.db["document_metadata"]
        self.chat_col = self.db["chat_history"]
        self.meta_col.create_index("weaviate_id", unique=True)

    def upsert_metadata(self, weaviate_id: str, metadata: Dict[str, Any]) -> None:
        """
        Insert or update the metadata document for a given weaviate_id (UUID).
        """
        doc = {"weaviate_id": weaviate_id, "metadata": metadata}
        # upsert: if exists, replace; if not, insert
        self.meta_col.replace_one({"weaviate_id": weaviate_id}, doc, upsert=True)

    def get_metadata(self, weaviate_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata by its weaviate UUID.
        Returns None if not found.
        """
        return self.meta_col.find_one({"weaviate_id": weaviate_id})

    #  Chat‑history helpers
    def get_chat_history(self,session_id,history_limit) -> str:
        """Return the last N turns (user + assistant) as a plain string."""
        cursor = (
            self.chat_col.find({"session_id": session_id})
            .sort("timestamp", -1)
            .limit(history_limit * 2)
        )
        # reverse to chronological order
        turns = list(cursor)[::-1]
        formatted = [
            f"{doc['role'].capitalize()}: {doc['message']}" for doc in turns
        ]
        return "\n".join(formatted) or "None yet."

    def store_chat(self, question: str, answer: str,session_id) -> None:
        """Persist both user question and assistant answer."""
        ts = datetime.datetime.utcnow()
        self.chat_col.insert_many(
            [
                {
                    "session_id": session_id,
                    "role": "user",
                    "message": question,
                    "timestamp": ts,
                },
                {
                    "session_id": session_id,
                    "role": "assistant",
                    "message": answer,
                    "timestamp": ts,
                },
            ]
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
