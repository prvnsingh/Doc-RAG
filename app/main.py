from settings import settings
import uvicorn
from fastapi import FastAPI, UploadFile, File
from services.extractor import Extractor
from services.vectorizerV2 import VectorizerV2
from services.summarizer import Summarizer
from services.retrieverV2 import RetrieverV2

from time import sleep
# Application settings
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
    # lifespan=lifespan,
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

global vector_retriever

@app.post("/upload_file_for_embedding")
def embedding_file(file: UploadFile = File(...)):
    # print(f' embedding file {file.filename}')
    file_content = file.file
    extractor = Extractor()
    extractor.run(file_content)
    summarizer = Summarizer(extractor.texts,extractor.tables,extractor.images_b64)
    summarizer.run()
    vectorizer = VectorizerV2()
    global vector_retriever
    vector_retriever = vectorizer.run(summarizer.texts,summarizer.tables,summarizer.images,
                            summarizer.text_summaries,summarizer.table_summaries,summarizer.image_summaries)
    return {"status": "success"}


@app.get("/ask_question")
def query_from_user(question:str):
    retriever = RetrieverV2(vector_retriever)
    response = retriever.run(question)
    return response

if __name__ == '__main__':
    uvicorn.run(app, port=settings.port, host=settings.host)