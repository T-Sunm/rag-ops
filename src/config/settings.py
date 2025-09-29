from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Any, Optional, Dict
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment support."""

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_STR: str = "/v1"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # Litellm Configuration
    LITELLM_BASE_URL: str = "http://localhost:4000"
    LITELLM_API_KEY: str = "sk-llmops"
    LITELLM_MODEL: str = os.getenv(
        "LITELLM_MODEL", "groq"
    )  # Default to groq if not set

    # LLM Parameters
    LLMs_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_STREAMING: bool = False

    # RAG Configuration - Dynamic dataset support
    DATASET_NAME: str = os.getenv(
        "DATASET_NAME", "environment_battery"
    )  # Dynamic from environment
    CHROMA_COLLECTION_NAME: str = f"rag-pipeline-{DATASET_NAME}"
    CHROMA_PERSIST_DIR: str = str(
        PROJECT_ROOT / "infrastructure" / "storage" / "chromadb"
    )

    # Performance & Caching
    CACHE_TTL: int = 3600
    MAX_RESPONSE_LENGTH: int = 2048
    REDIS_URI: str = "localhost:6378"

    # Langfuse Configuration
    LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_HOST: Optional[str] = os.getenv("LANGFUSE_HOST")

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration based on provider."""
        return {
            "temperature": self.LLMs_TEMPERATURE,
            "streaming": self.LLM_STREAMING,
            "max_tokens": self.LLM_MAX_TOKENS,
            "base_url": self.LITELLM_BASE_URL,
            "api_key": self.LITELLM_API_KEY,
            "model": self.LITELLM_MODEL,
        }


SETTINGS = Settings()

APP_CONFIGS: Dict[str, Any] = {
    "title": "RAG Ops - Production Architecture",
    "description": "Architecture RAG system with multi-LLM provider support via LiteLLM",
    "version": "1.0.0",
    "debug": SETTINGS.DEBUG,
}
