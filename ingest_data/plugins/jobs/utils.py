import os
import logging
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def check_src_data(file_link):
    return os.path.exists(file_link)


logger = get_logger()


class MinioLoader:
    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        secure: bool = False,
    ):
        self.client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=secure,
        )

    @staticmethod
    def get_info_from_minio(s3_path: str):
        s3_path = s3_path.replace("s3://", "")
        s3_bucket, s3_key = s3_path.split("/", 1)
        return s3_bucket, s3_key

    def upload_object_from_stream(
        self, s3_path: str, data_stream: BytesIO, data_length: int
    ):
        """Upload an object from a stream of bytes to MinIO."""
        s3_bucket, s3_key = self.get_info_from_minio(s3_path)
        if not self.client.bucket_exists(s3_bucket):
            self.client.make_bucket(s3_bucket)

        try:
            self.client.put_object(
                bucket_name=s3_bucket,
                object_name=s3_key,
                data=data_stream,
                length=data_length,
                content_type="application/octet-stream",
            )
            logger.info(f"Successfully uploaded data to '{s3_path}'")
        except S3Error as e:
            logger.error(f"Failed to upload to MinIO: {e}")
            raise

    def download_object_as_stream(self, s3_path: str) -> BytesIO:
        """Download an object as a stream of bytes from MinIO."""
        s3_bucket, s3_key = self.get_info_from_minio(s3_path)
        try:
            response = self.client.get_object(bucket_name=s3_bucket, object_name=s3_key)
            buffer = BytesIO(response.read())
            buffer.seek(0)
            logger.info(f"Successfully downloaded data from '{s3_path}'")
            return buffer
        except S3Error as e:
            logger.error(f"Failed to download from MinIO: {e}")
            raise


model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def get_tokenizer():
    print(f"🔄 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device.upper()} ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )
    return embeddings
