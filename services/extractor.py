from components.base_component import BaseComponent
from unstructured.partition.pdf import partition_pdf

class Extractor(BaseComponent):
    def __init__(self):
        super().__init__('Extractor')
        self.texts = []
        self.tables = []
        self.images_b64 = []

    def run(self, pdf_data, output_dir="resources/extracted_content"):

        chunks = partition_pdf(
        file=pdf_data,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image","Table"],   # Add 'Table' to list to extract image of tables
        image_output_dir_path=output_dir,   # if None, images and tables will saved in base64

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="basic",          # or 'basic'
        max_characters=5000,                  # defaults to 500
        combine_text_under_n_chars=1000,       # defaults to 0
        new_after_n_chars=3000,
        )

    #     chunksV2 = partition_pdf(
    #     file=pdf_data,
    #     strategy="ocr_only",  # Forces Tesseract OCR use
    #     extract_image_block_types=["Image", "Table"],
    #     extract_image_block_to_payload=True,
    #     image_output_dir_path="resources/extracted_content",
    #     infer_table_structure=False,  # Not available in OCR-only
    # )

        
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                self.tables.append(chunk)

            if "CompositeElement" in str(type(chunk)):
                self.texts.append(chunk)
            
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        self.images_b64.append(el.metadata.image_base64)


        self.logger.info(f'Extracted texts = {len(self.texts)} tables= {len(self.tables)} images = {len(self.images_b64)}')