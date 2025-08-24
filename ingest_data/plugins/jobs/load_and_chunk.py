from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from plugins.jobs.utils import Minio_Loader

from plugins.config.minio_config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
)

minio_loader = Minio_Loader(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)


def remove_non_utf8_characters(text: str) -> str:
    return "".join(char for char in text if ord(char) < 128)


def load_pdf(pdf_file: str):
    docs = PyPDFLoader(pdf_file, extract_images=False).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs


def load_word_document(word_file: str):
    """Load a Word document (.docx or .doc) and return cleaned documents."""
    docs = Docx2txtLoader(word_file).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs


def get_num_cpu() -> int:
    return multiprocessing.cpu_count()


class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        raise NotImplementedError


class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str]):
        doc_loaded = []
        total_files = len(pdf_files)
        for pdf_file in tqdm(
            pdf_files, total=total_files, desc="Loading PDFs", unit="file"
        ):
            result = load_pdf(pdf_file)
            doc_loaded.extend(result)
        return doc_loaded


class WordDocumentLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, word_files: List[str]):
        doc_loaded = []
        total_files = len(word_files)
        for word_file in tqdm(
            word_files, total=total_files, desc="Loading Word documents", unit="file"
        ):
            result = load_word_document(word_file)
            doc_loaded.extend(result)
        return doc_loaded


class TextSplitter:
    def __init__(
        self,
        separators: List[str] = ["\n\n", "\n", " ", ""],
        chunk_size: int = 300,
        chunk_overlap: int = 13,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)


# số lượng từ trung bình trong 1 dòng là 13
class LoadAndChunk:
    def __init__(
        self,
        file_type: Literal["pdf", "docx", "auto"] = "auto",
        split_kwargs: dict = {"chunk_size": 300, "chunk_overlap": 13},
    ) -> None:
        self.file_type = file_type
        self.pdf_loader = PDFLoader()
        self.word_loader = WordDocumentLoader()
        self.doc_splitter = TextSplitter(**split_kwargs)

    def read_and_chunk(self, files: Union[str, List[str]]):
        if isinstance(files, str):
            files = [files]

        docs = []

        # Separate files by type
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]
        word_files = [f for f in files if f.lower().endswith((".docx", ".doc"))]

        # Load PDF files
        if pdf_files:
            pdf_docs = self.pdf_loader(pdf_files)
            docs.extend(pdf_docs)

        # Load Word files
        if word_files:
            word_docs = self.word_loader(word_files)
            docs.extend(word_docs)

        if not docs:
            raise ValueError(
                f"No supported files found. Supported formats: PDF, DOCX, DOC"
            )

        return self.doc_splitter(docs)

    def ingest_to_minio(self, data, s3_path: str):
        minio_loader.upload_to_minio(data, s3_path)

    def load_from_minio(self, s3_path: str):
        data = minio_loader.download_from_minio(s3_path)
        return data

    def load_dir(self, dir_path: str):
        # Support both PDF and Word files
        pdf_files = glob.glob(f"{dir_path}/*.pdf")
        word_files = glob.glob(f"{dir_path}/*.docx") + glob.glob(f"{dir_path}/*.doc")

        all_files = pdf_files + word_files

        if not all_files:
            raise ValueError(f"No PDF or Word document files found in {dir_path}")

        print(
            f"Found {len(pdf_files)} PDF files and {len(word_files)} Word document files"
        )
        return all_files
