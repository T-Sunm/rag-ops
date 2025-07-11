from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from plugins.jobs.utils import Minio_Loader

from plugins.config.minio_config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
minio_loader = Minio_Loader(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)


def remove_non_utf8_characters(text: str) -> str:
    return ''.join(char for char in text if ord(char) < 128)


def load_pdf(pdf_file: str):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
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
        for pdf_file in tqdm(pdf_files, total=total_files, desc="Loading PDFs", unit="file"):
            result = load_pdf(pdf_file)
            doc_loaded.extend(result)
        return doc_loaded


class TextSplitter:
    def __init__(
        self,
        separators: List[str] = ['\n\n', '\n', ' ', ''],
        chunk_size: int = 300,
        chunk_overlap: int = 13
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
        file_type: Literal["pdf"] = "pdf",
        split_kwargs: dict = {"chunk_size": 300, "chunk_overlap": 13}
    ) -> None:
        assert file_type == "pdf", "file_type must be pdf"
        self.file_type = file_type
        self.doc_loader = PDFLoader()
        self.doc_splitter = TextSplitter(**split_kwargs)

    def read_and_chunk(self, pdf_files: Union[str, List[str]]):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        docs = self.doc_loader(pdf_files)
        return self.doc_splitter(docs)
    

    def ingest_to_minio(self, data, s3_path: str):
        minio_loader.upload_to_minio(data, s3_path)

    def load_from_minio(self, s3_path: str):
        data = minio_loader.download_from_minio(s3_path)
        return data
    
    def load_dir(self, dir_path: str):
        files = glob.glob(f"{dir_path}/*.pdf")
        assert files, f"No PDF files found in {dir_path}"
        return files
