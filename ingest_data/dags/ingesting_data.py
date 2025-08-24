from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import wget
import sys
from pathlib import Path
from airflow.decorators import task
import chromadb
from airflow.utils.trigger_rule import TriggerRule

# Add plugins directory to Python path
AIRFLOW_HOME = Path("/opt/airflow")
sys.path.append(str(AIRFLOW_HOME))
from plugins.jobs.download import DATASETS, get_dataset_names
from plugins.jobs.utils import check_src_data
from plugins.jobs.load_and_chunk import LoadAndChunk
from plugins.jobs.embed_and_store import EmbedAndStore
from airflow.operators.empty import EmptyOperator

# Configuration - có thể set qua Airflow Variables
DATASET_NAME = os.getenv("DATASET_NAME", "environment_battery")  # Default dataset
dataset_folder = os.getenv("INLINE_DATA_VOLUME")
directory_chromadb = os.getenv("PERSIST_DIRECTORY")

# Dynamic naming based on dataset
MINIO_PATH = f"rag-pipeline-{DATASET_NAME}/chunks.pkl"
collection_name = f"rag-pipeline-{DATASET_NAME}"
dataset_subfolder = os.path.join(dataset_folder, DATASET_NAME)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


@task()
def start_task():
    """Downloads papers if they don't exist in the dataset folder."""
    folder_path = str(dataset_subfolder)
    os.makedirs(folder_path, exist_ok=True)

    dataset_files = DATASETS[DATASET_NAME]["data"]

    for file_link in dataset_files:
        dest_file_path = os.path.join(folder_path, f"{file_link['title']}.pdf")
        if not check_src_data(dest_file_path):
            print(f"Downloading {file_link['title']}...")
            wget.download(file_link["url"], out=dest_file_path)
        else:
            print(f"File {file_link['title']}.pdf already exists, skipping download.")

    return {"status": "completed", "folder_path": folder_path, "dataset": DATASET_NAME}


@task.branch()
def check_collection_task(data):
    """
    Checks if the ChromaDB collection exists and branches accordingly.
    """
    try:
        client = chromadb.PersistentClient(path=directory_chromadb)
        client.get_collection(name=collection_name)
        return "class_already_exists"
    except ValueError:
        return "create_class"


@task()
def create_class():
    """Creates a new class in ChromaDB."""
    print("Creating a new class in ChromaDB...")
    return True  # Indicate that the class was created


@task()
def class_already_exists():
    """Handles the case when the class already exists in ChromaDB."""
    print("Class already exists in ChromaDB.")
    return False  # Indicate that no class was created


@task()
def load_and_chunk_data():
    loader = LoadAndChunk()
    pdf_files = loader.load_dir(dataset_subfolder)  # Use subfolder
    chunks = loader.read_and_chunk(pdf_files)
    loader.ingest_to_minio(chunks, MINIO_PATH)  # Dynamic MINIO path
    return {"status": "completed"}


@task(trigger_rule=TriggerRule.ONE_SUCCESS)
def embed_and_store_data():
    embedder = EmbedAndStore()
    splits = embedder.minio_loader.download_from_minio(MINIO_PATH)
    vectordb = embedder.document_embedding_vectorstore(
        splits, collection_name, directory_chromadb
    )  # Dynamic collection name
    return {"status": "completed"}


# Create DAG
with DAG(
    "ingest_data",
    default_args=default_args,
    description="A DAG to ingest data",
    schedule_interval=None,
) as dag:

    # Tasks
    start = start_task()
    branch = check_collection_task(start)
    create = create_class()
    exists = class_already_exists()
    load_chunk = load_and_chunk_data()
    embed_store = embed_and_store_data()
    end_task = EmptyOperator(task_id="end_task")
    # Task flow
    start >> branch
    branch >> [create, exists]  # Branching
    create >> load_chunk >> embed_store  # Process path
    exists >> embed_store  # Skip path
    embed_store >> end_task
