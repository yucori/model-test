from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434"
    port: int = 8765
    upload_dir: str = "./uploads"
    chroma_data_dir: str = "./chroma_data"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
