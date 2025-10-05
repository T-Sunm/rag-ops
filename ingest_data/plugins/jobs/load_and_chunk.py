import pickle
from typing import Union, List, Optional
import glob
from tqdm import tqdm
import multiprocessing
from io import BytesIO

# Docling imports
from langchain_docling.loader import DoclingLoader, ExportType
from docling.chunking import HybridChunker
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from langchain_text_splitters import RecursiveCharacterTextSplitter
from plugins.jobs.utils import get_tokenizer

from plugins.jobs.utils import MinioLoader
from plugins.config.minio_config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
)

minio_loader = MinioLoader(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)


def get_num_cpu() -> int:
    return multiprocessing.cpu_count()


def create_advanced_converter():
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = False
    pdf_pipeline_options.do_table_structure = True  # Bật nhận dạng cấu trúc bảng
    pdf_pipeline_options.table_structure_options.do_cell_matching = True
    # Sử dụng chế độ chính xác cao nhất để nhận dạng bảng
    pdf_pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # Tạo converter hỗ trợ cả PDF và DOCX
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options, backend=PyPdfiumDocumentBackend
            ),
            # Support cho DOCX - Docling tự động xử lý DOCX tốt với default settings
            InputFormat.DOCX: None,
        }
    )

    print("-> Converter đã được cấu hình cho PDF và DOCX")
    return doc_converter


class LoadAndChunk:
    def __init__(
        self,
        embed_model_id: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        max_tokens: int = 512,
        chunk_overlap: int = 50,
        split_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Khởi tạo LoadAndChunk với Docling.

        Args:
            embed_model_id: Model embedding để tính toán token
            max_tokens: Số token tối đa cho mỗi chunk
            chunk_overlap: Số token gối đầu giữa các chunk
            models_cache_dir: Thư mục cache cho các model Docling
            split_kwargs: Tham số legacy (giữ lại để tương thích)
        """
        self.embed_model_id = embed_model_id
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.num_processes = get_num_cpu()

        # Lazy initialization
        self.converter = None
        self.tokenizer = None
        self.recursive_splitter = None

    def _init_converter(self):
        """Lazy initialization of converter"""
        if self.converter is None:
            self.converter = create_advanced_converter()

    def _init_tokenizer_and_splitter(self):
        """Lazy initialization of tokenizer and splitter"""
        if self.tokenizer is None:
            print(
                "-> Đang khởi tạo tokenizer cho model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
            self.tokenizer = get_tokenizer()

            self.recursive_splitter = (
                RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    tokenizer=self.tokenizer,
                    chunk_size=self.max_tokens,
                    chunk_overlap=self.chunk_overlap,
                )
            )

    def read_and_chunk(self, files: Union[str, List[str]]):
        """
        Đọc và chia nhỏ tài liệu sử dụng Docling.

        Args:
            files: Đường dẫn file hoặc danh sách đường dẫn file

        Returns:
            List của các Document đã được chia nhỏ
        """
        if isinstance(files, str):
            files = [files]

        # Khởi tạo converter và tokenizer
        self._init_converter()
        self._init_tokenizer_and_splitter()

        # Filter files that Docling can handle
        supported_files = [
            f for f in files if f.lower().endswith((".pdf", ".docx", ".doc"))
        ]

        if not supported_files:
            raise ValueError(
                "No files supported by Docling found. Supported formats: PDF, DOCX, DOC"
            )

        if len(supported_files) != len(files):
            unsupported = [f for f in files if f not in supported_files]
            print(
                f"Warning: {len(unsupported)} files not supported by Docling will be skipped: {unsupported}"
            )

        print(f"Processing {len(supported_files)} files with Docling...")

        all_docs = []

        for file_path in tqdm(
            supported_files, desc="Processing files with Docling", unit="file"
        ):
            print(f"\n-> Bắt đầu đọc và chunking tài liệu: {file_path}")

            # Khởi tạo DoclingLoader với converter đã tùy chỉnh
            loader = DoclingLoader(
                file_path=[file_path],  # DoclingLoader expects a list
                export_type=ExportType.DOC_CHUNKS,
                converter=self.converter,
                chunker=HybridChunker(tokenizer=self.embed_model_id),
            )

            # Lấy các chunk ban đầu từ Docling
            initial_docs = loader.load()
            print(f"==> Số chunk ban đầu từ Docling: {len(initial_docs)}")

            # Xử lý hậu kỳ để đảm bảo các chunk không vượt quá max_tokens
            print(
                f"-> Bắt đầu xử lý hậu kỳ để đảm bảo các chunk không vượt quá {self.max_tokens} token..."
            )

            final_splits = []
            oversized_chunks_count = 0

            for doc in initial_docs:
                # Đếm số token trong chunk hiện tại
                num_tokens = len(
                    self.tokenizer.encode(doc.page_content, add_special_tokens=False)
                )

                if num_tokens > self.max_tokens:
                    oversized_chunks_count += 1
                    # Nếu chunk quá lớn, dùng recursive_splitter để chia nhỏ nó ra
                    sub_splits = self.recursive_splitter.split_documents([doc])
                    final_splits.extend(sub_splits)
                else:
                    # Nếu chunk có kích thước ổn, giữ nguyên nó
                    final_splits.append(doc)

            print(
                f"==> Đã phát hiện và chia nhỏ {oversized_chunks_count} chunk quá khổ."
            )
            print(f"==> Tổng số chunk cuối cùng sau khi xử lý: {len(final_splits)}")

            all_docs.extend(final_splits)

        return all_docs

    def ingest_to_minio(self, data, s3_path: str):
        """
        Serialize dữ liệu bằng pickle và upload lên MinIO.
        Hàm này giờ sẽ chịu trách nhiệm cho việc chuẩn bị dữ liệu.
        """
        print(f"-> Bắt đầu serialize và ingest dữ liệu tới: {s3_path}")

        buffer = BytesIO()
        pickle.dump(data, buffer)
        data_length = buffer.tell()
        buffer.seek(0)  # Đưa con trỏ về đầu để MinioLoader có thể đọc

        minio_loader.upload_object_from_stream(
            s3_path=s3_path, data_stream=buffer, data_length=data_length
        )
        print("-> Ingest thành công!")

    def load_from_minio(self, s3_path: str):
        """
        Download dữ liệu từ MinIO và deserialize bằng pickle.
        """
        print(f"-> Bắt đầu download và deserialize dữ liệu từ: {s3_path}")

        buffer = minio_loader.download_object_as_stream(s3_path)

        data = pickle.load(buffer)
        print("-> Download và deserialize thành công!")
        return data

    def load_dir(self, dir_path: str):
        """
        Tìm tất cả file PDF và Word trong thư mục.

        Args:
            dir_path: Đường dẫn thư mục

        Returns:
            List đường dẫn file
        """
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

    def process_directory(self, dir_path: str):
        """
        Xử lý toàn bộ thư mục: tìm file và chia nhỏ.

        Args:
            dir_path: Đường dẫn thư mục

        Returns:
            List của các Document đã được chia nhỏ
        """
        files = self.load_dir(dir_path)
        return self.read_and_chunk(files)
