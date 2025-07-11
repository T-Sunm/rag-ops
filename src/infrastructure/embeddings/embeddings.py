from sentence_transformers import SentenceTransformer
from typing import List
from langchain.embeddings.base import Embeddings
import numpy as np

class EmbeddingService(Embeddings):
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.embedding_model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single text (normalized vector) and return as list."""
        # SentenceTransformer.encode returns a numpy array for a single string
        vector = self.embedding_model.encode(text, normalize_embeddings=True)
        return np.array(vector).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (normalized vectors) and return as list of lists."""
        vectors = self.embedding_model.encode(texts, normalize_embeddings=True)
        return np.array(vectors).tolist()
