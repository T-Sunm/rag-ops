
from airflow.models import Variable
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = Variable.get("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = Variable.get("MINIO_SECRET_KEY")