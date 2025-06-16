"""
PDF content extraction service.
This module provides functionality to extract text, tables, and images from PDF documents
using the unstructured library with high-resolution processing capabilities.
"""

from components.base_component import BaseComponent
from unstructured.partition.pdf import partition_pdf
from .image_processing import filter_non_blank_images


class Extractor(BaseComponent):
    """
    PDF content extractor that processes PDF files to extract text, tables, and images.
    
    This class uses the unstructured library with high-resolution processing to extract
    various types of content from PDF documents, including:
    - Text content
    - Tables with structure preservation
    - Images and figures
    - Equations
    
    Attributes:
        texts (list): List of extracted text chunks
        tables (list): List of extracted tables
        images_b64 (list): List of base64-encoded images
    """

    def __init__(self):
        """Initialize the extractor with empty content lists."""
        super().__init__('Extractor')
        self.texts = []
        self.tables = []
        self.images_b64 = []

    def run(self, pdf_data, output_dir="resources/extracted_content"):
        """
        Process a PDF file and extract its contents.
        
        This method performs the following steps:
        1. Partition the PDF using high-resolution strategy
        2. Extract text, tables, and images
        3. Filter out blank images
        4. Store results in class attributes
        
        Args:
            pdf_data: The PDF file data to process
            output_dir (str): Directory to save extracted images (default: "resources/extracted_content")
            
        Note:
            The extraction process uses high-resolution processing to ensure accurate
            table structure inference and image extraction.
        """
        # Partition PDF with high-resolution processing
        chunks = partition_pdf(
            file=pdf_data,
            strategy="hi_res",  # mandatory to infer tables
            extract_images_in_pdf=True,
            infer_table_structure=True,  # extract tables
            extract_image_block_types=['Table', 'Figure'],
            include_page_breaks=True,
            unique_element_ids=True,
            # hi_res_model_name='yolox',
            image_output_dir_path=output_dir,  # if None, images and tables will saved in base64
            extract_image_block_to_payload=True,  # if true, will extract base64 for API usage
            chunking_strategy="by_title",  # or 'basic'
            max_characters=10000,  # defaults to 500
            combine_text_under_n_chars=2000,  # defaults to 0 if chunks substantially smaller than desired combine
            new_after_n_chars=6000,
            chunk_overlap=100,
        )

        # Process extracted chunks
        for chunk in chunks:
            chunk_dic = chunk.to_dict()
            self.logger.info(f'\n{chunk_dic.items()}\n')
            # Extract tables
            if 'Table' in str(type(chunk)) or 'TableChunk' in str(type(chunk)):
                self.tables.append({"text": chunk.metadata.text_as_html, "metadata": chunk_dic['metadata']})

            # Extract text
            if 'CompositeElement' in str(type(chunk)):
                self.texts.append({"text": chunk_dic['text'], "metadata": chunk_dic['metadata']})

            # Extract images
            if 'CompositeElement' in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        self.images_b64.append({"image": el.metadata.image_base64, "metadata": chunk_dic['metadata']})

        # Log extraction results
        self.logger.info(
            f'Extracted texts = {len(self.texts)} tables= {len(self.tables)} images = {len(self.images_b64)}')

        # Filter out blank images
        # self.images_b64 = filter_non_blank_images(self.images_b64)
        self.logger.info(
            f'Extracted texts = {len(self.texts)} tables= {len(self.tables)} images = {len(self.images_b64)}')
