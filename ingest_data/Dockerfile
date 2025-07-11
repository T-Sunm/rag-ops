FROM apache/airflow:2.10.3

# Set up C++11 for chromadb and rapidocr- PyPDFLoader (libgl1)
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Install Python packages
ADD requirements.txt .
RUN pip install --no-cache-dir apache-airflow==${AIRFLOW_VERSION} -r requirements.txt