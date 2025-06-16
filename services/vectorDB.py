from weaviate.classes.config import Configure
import weaviate
from app.config import config
from components.base_component import BaseComponent
from services.document_store import DocumentStore


class VectorDB(BaseComponent):
    def __init__(self):
        super().__init__('VectorDB')

        self.headers = {
            "X-AWS-Access-Key": config.AWS_ACCESS_KEY_ID,
            "X-AWS-Secret-Key": config.AWS_SECRET_ACCESS_KEY,
        }
        self.client = weaviate.connect_to_local(host=config.WEAVIATE_HOST, port=8080, grpc_port=50051,
                                                headers=self.headers)

    def run(self, data,app):
        self.logger.info(self.client.is_ready())
        # Check if class already exists
        existing_collections = self.client.collections.list_all()
        class_name = "DocumentCollection"
        if class_name not in existing_collections.keys():
            self.client.collections.create(
                class_name,
                vectorizer_config=[
                    Configure.NamedVectors.text2vec_aws(
                        name="text_vector",
                        region=config.AWS_REGION,
                        source_properties=["text"],
                        service="bedrock",
                        model="amazon.titan-embed-text-v2:0",
                    )
                ],
            )

        collection = self.client.collections.get(class_name)

        for data_chunk in data:
            uuid = collection.data.insert(
                properties={"text": data_chunk["text"]},
            )

            # population the vector store with the textual data and keeping the metadata for docstore
            doc_store = app.state.doc_store
            if 'image' in data_chunk.keys():
                doc = {k: data_chunk[k] for k in ("image", "metadata")}
                doc_store.upsert_metadata(str(uuid), doc)
            else:
                doc_store.upsert_metadata(str(uuid), data_chunk["metadata"])
            self.logger.info(f"Inserted meta data for document with UUID: {uuid}")