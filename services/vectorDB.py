from weaviate.classes.config import Configure
import weaviate
from weaviate.classes.config import Property, DataType
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
        self.doc_store = {}

    def run(self, data):
        with weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051, headers=self.headers) as client:
            self.logger.info(client.is_ready())
            # Check if class already exists
            existing_collections = client.collections.list_all()
            class_name = "DocumentCollection"
            if class_name not in existing_collections.keys():
                client.collections.create(
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

            collection = client.collections.get(class_name)

            for data_chunk in data:
                uuid = collection.data.insert(
                    properties={"text": data_chunk["text"]},
                )

                # population the vector store with the textual data and keeping the metadata for docstore
                with DocumentStore(config.MONGO_URI) as doc_store:
                    self.doc_store[uuid] = data_chunk['metadata']
                    if 'image' in data_chunk.keys():
                        doc = {k: data_chunk[k] for k in ("image", "metadata")}
                        doc_store.upsert_metadata(str(uuid), doc)
                    else:
                        doc_store.upsert_metadata(str(uuid), data_chunk["metadata"])
        return self.doc_store
