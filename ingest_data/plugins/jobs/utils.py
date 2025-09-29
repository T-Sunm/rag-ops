import chromadb
import os
import logging
import pickle
from io import BytesIO
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from typing import Optional
from sentence_transformers import SentenceTransformer
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


class Minio_Loader:
    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        secure: bool = False,
    ):
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.secure = secure
        self.client = Minio(
            self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            secure=self.secure,
        )

    @staticmethod
    def get_info_from_minio(s3_path: str):
        s3_path = s3_path.replace("s3://", "")  # loại bỏ s3:// nếu có
        s3_bucket, s3_key = s3_path.split(
            "/", 1
        )  # tách bucket và key -- s3_path = "bucket/key"
        return s3_bucket, s3_key

    def upload_to_minio(self, data, s3_path: str):
        s3_bucket, s3_key = self.get_info_from_minio(s3_path)
        logger.info(f"Bucket: {s3_bucket}")
        logger.info(f"Key: {s3_key}")

        if not self.client.bucket_exists(s3_bucket):
            self.client.make_bucket(s3_bucket)
            logger.info(f"Created bucket '{s3_bucket}'")
        else:
            logger.info(f"Bucket '{s3_bucket}' already exists")

        buffer = BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        try:
            self.client.put_object(
                bucket_name=s3_bucket,
                object_name=s3_key,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type="application/octet-stream",
            )
            logger.info(f"Uploaded data to '{s3_path}'")
        except S3Error as e:
            logger.error(f"Failed to upload to MinIO: {e}")
            raise

    def download_from_minio(self, s3_path: str):
        s3_bucket, s3_key = self.get_info_from_minio(s3_path)
        buffer = BytesIO()

        try:
            response = self.client.get_object(bucket_name=s3_bucket, object_name=s3_key)
            for chunk in response.stream(32 * 1024):
                buffer.write(chunk)
            buffer.seek(0)
            obj = pickle.load(buffer)
            logger.info(f"Downloaded data from '{s3_path}'")
            return obj
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
