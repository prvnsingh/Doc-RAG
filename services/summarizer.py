"""
Content summarization service.
This module provides functionality to generate summaries of text, tables, and images
using a large language model (LLM) through the Bedrock service.
"""

from components.base_component import BaseComponent
from .bedrock import MLLM
from app.prompt import summary_prompt_text, summary_prompt_image


class Summarizer(BaseComponent):
    """
    Content summarizer that generates concise summaries of text, tables, and images.
    
    This class uses a large language model to create summaries of different types of content:
    - Text passages
    - Tables (converted to HTML format)
    - Images (with detailed descriptions)
    
    Attributes:
        texts (list): List of text chunks to summarize
        tables (list): List of tables to summarize
        images (list): List of images to summarize
        text_summaries (list): Generated summaries of text chunks
        image_summaries (list): Generated descriptions of images
        table_summaries (list): Generated summaries of tables
        model (MLLM): Instance of the language model for summarization
    """

    def __init__(self, texts, tables, images):
        """
        Initialize the summarizer with content to process.
        
        Args:
            texts (list): List of text chunks to summarize
            tables (list): List of tables to summarize
            images (list): List of images to summarize
        """
        super().__init__(logger_name='Summarizer')
        self.texts = texts
        self.tables = tables
        self.images = images
        self.text_summaries = []
        self.image_summaries = []
        self.table_summaries = []
        self.model = MLLM()

    def run(self):
        """
        Generate summaries for all content types.
        
        This method processes each type of content (text, tables, images) and generates
        appropriate summaries using the language model. The summaries are stored in
        the corresponding class attributes.
        
        The process includes:
        1. Summarizing text chunks using a text-specific prompt
        2. Converting tables to HTML and summarizing them
        3. Generating detailed descriptions of images using a specialized image prompt
        """
        # Summarize text chunks
        for text in self.texts:
            content = [{"type": "text", "text": summary_prompt_text.format(element=text)}]
            self.text_summaries.append(self.model.run(content))

        # Summarize tables (converted to HTML)
        for table in self.tables:
            content = [{"type": "text", "text": summary_prompt_text.format(element=table.metadata.text_as_html)}]
            self.table_summaries.append(self.model.run(content))

        # Generate image descriptions
        for image in self.images:
            # Prepare content with both text prompt and image data
            content = [{"type": "text", "text": summary_prompt_image}]
            content_dict = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image
                }
            }
            content.append(content_dict)
            self.image_summaries.append(self.model.run(content))

        # Log summary generation results
        self.logger.info(f'''summaries    text_summaries: {self.text_summaries}
                                            table_summaries: {self.table_summaries}
                                            image_summaries: {self.image_summaries}''')
                