import streamlit as st
import requests
from io import BytesIO
import base64
from PIL import Image
# Set page config
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("Document Q&A System")
st.markdown("""
This application allows you to:
1. Upload documents for processing
2. Ask questions about the uploaded documents
3. Get AI-powered responses
""")

# Initialize session state for tracking uploaded files
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# File upload section
st.header("📄 Upload Document")
uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt'])

if uploaded_file is not None:
    # Display file details
    st.write("File details:")
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size} bytes")
    
    # Process file button
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            # Prepare file for upload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            
            # Call the embedding endpoint
            response = requests.post("http://localhost:8000/upload_file_for_embedding", files=files)
            
            if response.status_code == 200:
                st.success("Document processed successfully!")
                st.session_state.file_uploaded = True
            else:
                st.error("Error processing document. Please try again.")

# Question section
st.header("❓ Ask Questions")
question = st.text_input("Enter your question about the document:")

if question and st.button("Get Answer"):
    with st.spinner("Getting answer..."):
        # Call the question endpoint
        response = requests.get(f"http://localhost:8000/ask_question?question={question}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Display the answer
            st.subheader("Answer:")
            st.write(result["answer"])
            
            # Display context texts if available
            if result.get("context_texts"):
                st.subheader("Relevant Context:")
                for i, text in enumerate(result["context_texts"], 1):
                    st.markdown(f"**Context {i}:**")
                    st.write(text)
            
            # Display context images if available
            if result.get("context_images"):
                st.subheader("Relevant Images:")
                cols = st.columns(min(3, len(result["context_images"])))
                for i, img_b64 in enumerate(result["context_images"]):
                    with cols[i % 3]:
                        image_bytes = base64.b64decode(img_b64)
                        image = Image.open(BytesIO(image_bytes))
                        st.image(image, caption=f"Image {i+1}")
        else:
            st.error("Error getting answer. Please try again.")

# Query decomposition section (for debugging/analysis)
st.header("🔍 Query Analysis")
if st.checkbox("Show query decomposition"):
    if question:
        with st.spinner("Analyzing query..."):
            response = requests.get(f"http://localhost:8000/query_decompose?question={question}")
            if response.status_code == 200:
                decomposition = response.json()
                st.subheader("Query Decomposition:")
                st.json(decomposition)
            else:
                st.error("Error analyzing query. Please try again.") 