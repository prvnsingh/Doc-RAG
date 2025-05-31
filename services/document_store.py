from pymongo import MongoClient
from typing import Any, Dict, Optional


class DocumentStore:
    def __init__(self, mongo_uri, db_name="my_db", collection_name="document_metadata"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.col = self.db[collection_name]
        self.col.create_index("weaviate_id", unique=True)

    def upsert_metadata(self, weaviate_id: str, metadata: Dict[str, Any]) -> None:
        """
        Insert or update the metadata document for a given weaviate_id (UUID).
        """
        doc = {"weaviate_id":weaviate_id, "metadata": metadata}
        # upsert: if exists, replace; if not, insert
        self.col.replace_one({"weaviate_id":  weaviate_id}, doc, upsert=True)

    def get_metadata(self, weaviate_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata by its weaviate UUID.
        Returns None if not found.
        """
        return self.col.find_one({"weaviate_id": weaviate_id})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()