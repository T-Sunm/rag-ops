from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Any, Optional, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment support."""

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_STR: str = "/v1"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # LLM Provider Selection
    LLM_PROVIDER: str = "lmstudio"  # lmstudio | openai | ollama

    # LM Studio Configuration
    LM_STUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    LM_STUDIO_API_KEY: str = "dummy"
    LM_STUDIO_MODEL: str = "qwen2.5-1.5b-instruct"

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"

    # LLM Parameters
    LLMs_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_STREAMING: bool = True

    # Langfuse Configuration
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_HOST: Optional[str] = "http://localhost:3000"

    # RAG Configuration
    CHROMA_COLLECTION_NAME: str = "rag-pipeline"
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "DATA" / "chromadb")

    # Performance & Caching
    CACHE_TTL: int = 3600
    MAX_RESPONSE_LENGTH: int = 2048
    REDIS_URI: str = "localhost:6378"

    class Config:
        env_file = ".env"

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration based on provider."""
        base_config = {
            "temperature": self.LLMs_TEMPERATURE,
            "streaming": self.LLM_STREAMING,
            "max_tokens": self.LLM_MAX_TOKENS,
        }

        if self.LLM_PROVIDER == "lmstudio":
            return {
                **base_config,
                "openai_api_base": self.LM_STUDIO_BASE_URL,
                "openai_api_key": self.LM_STUDIO_API_KEY,
                "model": self.LM_STUDIO_MODEL,
            }
        elif self.LLM_PROVIDER == "openai":
            if not self.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is required when using OpenAI provider"
                )
            return {
                **base_config,
                "openai_api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL,
            }
        elif self.LLM_PROVIDER == "ollama":
            return {
                **base_config,
                "openai_api_base": self.OLLAMA_BASE_URL,
                "openai_api_key": "dummy",
                "model": self.OLLAMA_MODEL,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.LLM_PROVIDER}")


SETTINGS = Settings()

APP_CONFIGS: Dict[str, Any] = {
    "title": "RAG Ops - Production Architecture Demo",
    "description": "Clean architecture RAG system with multi-LLM provider support",
    "version": "1.0.0",
    "debug": SETTINGS.DEBUG,
}
