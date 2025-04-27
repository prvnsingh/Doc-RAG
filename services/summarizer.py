from components.base_component import BaseComponent
from .bedrock import MLLM
from app.prompt import summary_prompt_text,summary_prompt_image


class Summarizer(BaseComponent):

    def __init__(self,texts,tables,images):
        super().__init__(logger_name='Summarizer')
        self.texts = texts
        self.tables = tables
        self.images = images
        self.text_summaries = []
        self.image_summaries = []
        self.table_summaries = []
        self.model = MLLM()


    def run(self):
        for text in self.texts:
            content = [{"type": "text", "text": summary_prompt_text.format(element = text)}]
            self.text_summaries.append(self.model.run(content)) 

        for table in self.tables:
            content = [{"type": "text", "text": summary_prompt_text.format(element = table.metadata.text_as_html)}]
            self.table_summaries.append(self.model.run(content))

        for image in self.images: # text prompt with chain of thoughts
            content = [{"type": "text", "text": summary_prompt_image}] 
            content_dict = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                            "data": image}}  # adding image to the text prompt
            content.append(content_dict)
            self.image_summaries.append(self.model.run(content))

        self.logger.info(f'''summaries    text_summaries: {self.text_summaries}
                                            table_summaries: {self.table_summaries}
                                            image_summaries: {self.image_summaries}''')
                