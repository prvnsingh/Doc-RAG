import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        #properties with default values
        self.AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        self.AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.AWS_REGION = os.getenv('AWS_REGION')
        self.IS_LOCAL = os.getenv('IS_LOCAL', "False")
        self.APP_VERSION = os.getenv('APP_VERSION', '0.0.0')
        self.OCR_AGENT=os.getenv('OCR_AGENT',"pytesseract")
        self.TESSERACT_LANGUAGE=os.getenv('TESSERACT_LANGUAGE','eng')
        self.UNSTRUCTURED_HI_RES_MODEL_NAME=os.getenv('UNSTRUCTURED_HI_RES_MODEL_NAME','yolox')
        # self.MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

    def get(self, tag):
        # Use Python's built-in `getattr` to dynamically access attributes by name
        return getattr(self, tag, None)

config = Config()
