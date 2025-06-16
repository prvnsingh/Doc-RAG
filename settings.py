"""
Configuration settings for the Multi-Modal RAG Application.
This module defines all application-wide settings and configurations using Pydantic for type safety.
"""

from pydantic_settings import BaseSettings
from app.config import config


class Settings(BaseSettings):
    """
    Application settings and configuration parameters.
    
    This class defines all configurable parameters for the application using Pydantic's
    BaseSettings for automatic environment variable loading and type validation.
    
    Attributes:
        app_name (str): Name of the application
        app_description (str): Detailed description of the application's purpose
        version (str): Current version of the application
        debug (bool): Debug mode flag
        port (int): Port number for the application server
        host (str): Host address for the application server
        admin_email (str): Administrator contact email
        MODEL_ID_SONNET_3_7 (str): Anthropic Claude 3.7 Sonnet model identifier
        ANTHROPIC_VERSION (str): Version of the Anthropic API being used
        MAX_TOKENS (int): Maximum number of tokens for model responses
        model_temp (float): Temperature parameter for model response generation
    """
    app_name: str = "Multi-Modal RAG Application"
    app_description: str = "This app is a QnA application what reads a pdf  and answer all your questions"
    version: str = config.get("APP_VERSION") or "1.0.0"
    debug: bool = False
    port: int = 8000
    host: str = "0.0.0.0"
    admin_email: str = "admin@example.com"
    MODEL_ID_SONNET_3_7: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"  # Sonnet3.7 model
    ANTHROPIC_VERSION: str = "bedrock-2023-05-31"
    MAX_TOKENS: int = 20000
    model_temp: float = 0.5
    persist_directory: str = "resources/chroma_langchain_db"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    IS_LOCAL:bool = True

# Create a global settings instance
settings = Settings()
