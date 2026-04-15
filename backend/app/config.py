from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    port: int = 8765
    upload_dir: str = "./uploads"
    chroma_data_dir: str = "./chroma_data"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
