from enum import Enum


class LLMModel(Enum):
    """
    Enum for LLM models.
    """
    OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    OPENAI_GPT_4O_MINI = "gpt-4o-mini"
    

class LLMProvider(Enum):
    """
    Enum for LLM providers.
    """
    OPENAI = "openai"