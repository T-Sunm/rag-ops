import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from uuid import uuid4
from plugins.jobs.utils import Minio_Loader
from plugins.config.minio_config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
)


class EmbedAndStore:
    def __init__(self):
        # Initialize the embedding model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Using device: {device.upper()} ---")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.minio_loader = Minio_Loader(
            MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        )

    def document_embedding_vectorstore(
        self, splits: list[Document], collection_name: str, persist_directory: str
    ):
        """
        Generate embeddings for document splits and store them in a Chroma vector store.

        Args:
            splits (list[Document]): List of Document objects with page_content.
            collection_name (str): Name of the Chroma collection.
            persist_directory (str): Local directory to persist the vector store.

        Returns:
            vectordb: The Chroma vector store instance.
        """
        print("========= Initializing Chroma Vector Store =============")

        # 1. Create or load the Chroma collection
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        # 2. Generate unique IDs for each document chunk
        uuids = [str(uuid4()) for _ in splits]

        # 3. Add documents to the vector store
        print(f"Adding {len(splits)} document chunks to vector store…")
        vectordb.add_documents(documents=splits, ids=uuids)

        return vectordb


# --------------------------- TEST -------------------------------
# if __name__ == "__main__":
#     # ---- Self-test ----

#     # 1. Create dummy document splits
#     dummy_texts = [
#         "Hello world, this is a test chunk.",
#         "Another chunk for embedding.",
#         "Final chunk to verify.",
#     ]
#     splits = [Document(page_content=t) for t in dummy_texts]

#     # 2. Setup a temporary directory for persistence
#     temp_dir = "./db"
#     collection_name = "test_embed_store"

#     # 3. Run embedding + store
#     es = EmbedAndStore()
#     vectordb = es.document_embedding_vectorstore(
#         splits=splits, collection_name=collection_name, persist_directory=temp_dir
#     )

#     # 4. Verify persistence directory is not empty
#     assert os.listdir(temp_dir), "Persist directory should not be empty"

#     # 5. Reload the store and check the number of embeddings
#     reloaded = Chroma(
#         collection_name=collection_name,
#         persist_directory=temp_dir,
#         embedding_function=HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2"
#         ),
#     )
#     count = reloaded._collection.count()
#     assert count == len(splits), f"Expected {len(splits)} embeddings, got {count}"

#     print(f"Self-test passed: {count} embeddings stored and reloaded successfully.")
#     print(
#         f"Temporary store is at: {temp_dir}\nPlease delete it manually when you're done."
#     )

#     query = "LangChain provides abstractions to make working with LLMs easy"
#     print(f"\nQuerying for top-2 results similar to:\n  {query!r}\n")

#     results = vectordb.similarity_search(
#         query,
#         k=2,
#         # filter={"source": "tweet"}   nếu bạn có metadata 'source'
#     )

#     for i, res in enumerate(results, 1):
#         print(f"{i}. {res.page_content!r}   --> metadata: {res.metadata}")

# python -m plugins.jobs.embed_and_store
