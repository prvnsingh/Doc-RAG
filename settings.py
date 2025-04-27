from pydantic_settings import BaseSettings
from app.config import config


class Settings(BaseSettings):
    """
    Application settings
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
    TEMP: float = 0.5

settings = Settings()
