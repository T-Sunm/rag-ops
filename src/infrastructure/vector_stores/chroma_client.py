from langchain_chroma import Chroma
from langfuse import observe
from src.infrastructure.embeddings.embeddings import embedding_service
from src.config.settings import SETTINGS
from langchain.schema.document import Document
from typing import List, Tuple, Dict, Any
from src.cache.standard_cache import standard_cache

def _format_docs(
    docs: List[Document],
    scores: List[float] | None = None
) -> str:
    formatted = []
    for idx, doc in enumerate(docs):
        content = doc.page_content.strip()
        if scores:
            content += f"  [score={scores[idx]:.4f}]"
        formatted.append(content)
    return "\n\n".join(formatted)

class ChromaClientService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_service = embedding_service

    def _connect(self):
        persist_dir = SETTINGS.CHROMA_PERSIST_DIR

        self.client = Chroma(
            collection_name=SETTINGS.CHROMA_COLLECTION_NAME,
            persist_directory=str(persist_dir),
            embedding_function=self.embedding_service,
        )

    def retrieve_vector(
        self,
        question: str,
        top_k: int = 3,
        with_score: bool = False,
        metadata_filter: Dict[str, Any] = {}
    ) -> str:

        if self.client is None:
            self._connect()

        if with_score:
            docs_with_scores: List[Tuple[Document, float]] = (
                self.client.similarity_search_with_score(
                    question,
                    k=top_k,
                    filter=metadata_filter
                )
            )
            try:
                docs, scores = zip(*docs_with_scores)
                return _format_docs(list(docs), list(scores))
            except ValueError:
                return "Không tìm thấy tài liệu phù hợp."

        else:
            docs: List[Document] = self.client.similarity_search(
                question,
                k=top_k,
                filter=metadata_filter
            )
            return _format_docs(docs)
