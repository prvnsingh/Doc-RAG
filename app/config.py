"""
Configuration management module for the Multi-Modal RAG system.
This module handles loading and managing environment variables and application configuration.
"""

import os
from dotenv import load_dotenv

class Config:
    """
    Configuration manager class that loads and provides access to environment variables.
    
    This class is responsible for:
    1. Loading environment variables from .env file
    2. Providing default values for optional configurations
    3. Offering a method to access configuration values by name
    
    Attributes:
        AWS_ACCESS_KEY_ID (str): AWS access key for authentication
        AWS_SECRET_ACCESS_KEY (str): AWS secret key for authentication
        AWS_REGION (str): AWS region for service deployment
        IS_LOCAL (str): Flag indicating if running in local environment
        APP_VERSION (str): Current version of the application
        OCR_AGENT (str): OCR engine to use (default: pytesseract)
        TESSERACT_LANGUAGE (str): Language for Tesseract OCR (default: eng)
        UNSTRUCTURED_HI_RES_MODEL_NAME (str): Model name for high-resolution processing
    """
    
    def __init__(self):
        """Initialize configuration by loading environment variables."""
        load_dotenv()
        # Load required AWS credentials
        self.AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        self.AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.AWS_REGION = os.getenv('AWS_REGION')
        
        # Application settings with default values
        self.IS_LOCAL = os.getenv('IS_LOCAL', "False")
        self.APP_VERSION = os.getenv('APP_VERSION', '0.0.0')
        self.MONGO_URI = os.getenv('MONGO_URI'),
        self.NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY'),
        self.WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
        
        # self.MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

    def get(self, tag):
        """
        Get a configuration value by its name.
        
        Args:
            tag (str): Name of the configuration attribute to retrieve
            
        Returns:
            The value of the requested configuration, or None if not found
        """
        return getattr(self, tag, None)

# Create a global configuration instance
config = Config()
