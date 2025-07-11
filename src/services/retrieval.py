import json
from src.utils import logger


# Define the retrieval function
def retrieve(question: str, **kwargs):
    """Generate tool call for retrieval or respond."""

    # Perform the retrieval using the vector store
    # retrieved_docs = vector_store.similarity_search(state["question"])
    logger.info(f"Retrieving documents for question: {question}")
    retrieved_docs = [
        "Beetles are herbivores and eat plants, leaves, and flowers.",
        "Beetles are omnivores and eat a variety of foods, including other insects.",
        "Some beetles are scavengers and eat decaying organic matter.",
    ]
    return json.dumps(retrieved_docs)  # Return as JSON string for compatibility with LangChain tools
