# MRAG - Multimodal RAG PDF Question Answering

## Overview

MRAG (Multimodal Retrieval-Augmented Generation) is a powerful tool that enables users to upload any PDF document (books, user manuals, reports) and ask questions about its content. The system leverages multimodal processing to understand both text and visual elements within the document, providing comprehensive answers based on the entire context.

## Key Features

- **PDF Processing**: Upload any PDF document for analysis
- **Multimodal Understanding**: Processes both text and images within documents
- **Natural Language Querying**: Ask questions in plain English about any aspect of the document
- **Context-Aware Responses**: Receives answers that incorporate information from relevant sections
- **Interactive Web Interface**: User-friendly Streamlit interface for easy document upload and querying
- **Advanced Image Processing**: Enhanced image extraction and understanding using Pix2Text

## System Architecture

MRAG uses a "Summarization and Descriptive Embedding" approach where:

1. PDFs are preprocessed to extract text, images, and tables
2. A multimodal LLM (Claude 3.7 Sonnet) generates:
   - Summaries for extracted text
   - Detailed descriptions for extracted images
3. These summaries and descriptions are embedded into a single vector database
4. Each embedding is mapped to a unique ID linked to the original content
5. During inference, the system:
   - Vectorizes the user query
   - Performs similarity search to retrieve relevant content
   - Fetches the corresponding text/images using unique IDs
   - Generates comprehensive answers using the retrieved context

![System Architecture](MRAG-TAG.jpeg)

## Technology Stack

- **Backend**: Python with FastAPI
- **Frontend**: Streamlit for interactive web interface
- **MLLM**: Claude 3.7 Sonnet (via AWS Bedrock)
- **Vector Database**: ChromaDB
- **Key Libraries**:
  - Unstructured: For extracting elements from PDFs
  - LangChain: For implementing retrieval pipelines with ChromaDB
  - Pix2Text: For enhanced image and text extraction
  - FAISS: For efficient similarity search
  - LiteLLM: For LLM integration and management

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mrag.git
cd mrag

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your AWS credentials and other configuration
```

## Usage

You can run the application in two ways:

### 1. Streamlit Interface (Recommended)
```bash
# Start the Streamlit app
streamlit run app/streamlit_app.py

# Access the web interface
# Open your browser and go to http://localhost:8501
```

### 2. FastAPI Backend
```bash
# Start the FastAPI server
uvicorn app.main:app --reload

# Access the API
# Open your browser and go to http://localhost:8000
```

## Project Structure

```
src/
├── app/
│   ├── main.py           # FastAPI application
│   ├── streamlit_app.py  # Streamlit interface
│   ├── config.py         # Configuration settings
│   └── prompt.py         # Prompt templates
├── components/           # Reusable components
├── services/            # Core services
├── resources/           # Static resources
└── settings.py          # Project settings
```

## Future Improvements

1. Implement an ensembled extraction pipeline using tools like Pix2Text for more accurate extraction of images, tables, and mathematical equations
2. Introduce a dedicated datastore (MongoDB/PostgreSQL) to persist extracted content and metadata
3. Develop a better ranking mechanism using re-ranking models or hybrid retrieval techniques
4. Add support for more document formats beyond PDF
5. Implement user authentication and document management
6. Add batch processing capabilities for multiple documents

## Contributions

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
