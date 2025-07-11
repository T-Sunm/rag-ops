# RAG API Layer - Lab

This project provides a comprehensive API layer for a Retrieval-Augmented Generation (RAG) system. It includes data ingestion pipelines, a FastAPI-based application to serve the RAG model, and an observability stack using Langfuse.

## Architecture Components

- **API Service (`src/`)**: A FastAPI application that exposes multiple endpoints for interacting with the RAG system:
  - **REST API**: For standard request/response interactions.
  - **Server-Sent Events (SSE)**: For streaming responses back to the client.
  - **WebSocket**: For real-time, bidirectional communication.
- **Data Ingestion (`ingest_data/`)**: An Apache Airflow pipeline to process and ingest data into the vector store. The infrastructure includes:
  - **Airflow**: For orchestrating data workflows.
  - **PostgreSQL**: As the backend for Airflow.
  - **Redis**: As the message broker for Airflow.
  - **MinIO**: As an S3-compatible object store for data sources.
  - **ChromaDB**: As the vector store for ingested data (persisted in `DATA/chromadb`).
- **Observability (`observability/`)**: A dedicated stack for monitoring and tracing the RAG application, powered by **Langfuse**. This helps in tracking requests, responses, and the internal workings of the RAG pipeline.

## Setup and Installation

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [LM Studio](https://lmstudio.ai/) (optional, for local LLM serving)

### 1. Configure Environment

Create a `.env` file in the root of `lab-api-layer` by copying the `example.env`:

```bash
cp example.env .env
```

Now, edit the `.env` file and add your `OPENAI_API_KEY`.

### 2. Install Python Dependencies

Create a `conda` environment and install the required packages from `requirements.txt`.

```bash
# Create a new conda environment (e.g., named 'lab-api-layer' with Python 3.11)
conda create --name lab-api-layer python=3.10

# Activate the environment
conda activate lab-api-layer

# Install dependencies using pip
pip install -r requirements.txt
```

### 3. Setup LM Studio (Optional - for Local LLM)

If you want to use LM Studio instead of OpenAI API:

1. **Download and Install LM Studio** from [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Download a Model**: Use LM Studio to download your preferred model (e.g., Qwen, Llama, etc.)
3. **Start Local Server**:
   - Open LM Studio
   - Go to the "Server" tab
   - Load your downloaded model
   - Click "Start Server" (default port: 1234)
4. **Configure Langfuse Playground**:
   - Access Langfuse UI at http://localhost:3000
   - Go to Settings → LLM API Keys
   - Add new LLM API key with these settings:
     - **Provider name**: `lm-studio` (or any name you prefer)
     - **LLM adapter**: `openai`
     - **API Base URL**: `http://host.docker.internal:1234/v1`
     - **API Key**: `any-key-works` (LM Studio doesn't require real API key)
     - Enable "Enable default models"

> **Important**: Use `host.docker.internal:1234` instead of `localhost:1234` when running Langfuse in Docker, as it allows the container to access services on the host machine.

## How to Run the System

Follow these steps in order to get all the services running correctly.

### Step 1: Start Infrastructure Services

This project contains two separate `docker-compose` configurations for infrastructure.

#### A. Data Ingestion & Storage (Airflow, Minio, etc.)

This stack provides the services needed to run the data ingestion pipeline.

```bash
cd ingest_data/
docker compose up -d
```

#### B. Observability (Langfuse)

The Langfuse stack allows you to monitor and debug your RAG application.

```bash
cd observability/
docker compose up -d
```

### Step 2: Run the Data Ingestion Pipeline

Once the Airflow services are running, you need to trigger the DAG to process your documents.

1.  Navigate to the Airflow UI: **http://localhost:8080**
2.  Login with the default credentials (`airflow`/`airflow`).
3.  Find the DAG named `ingest_pipeline` (or similar) and trigger it manually.
4.  Wait for the DAG to complete successfully. This will process the PDFs in `DATA/data_source` and store the embeddings in ChromaDB.

### Step 3: Start the RAG API Server

After the infrastructure is up and the data has been ingested, you can start the FastAPI application.

```bash
# Make sure you have activated your conda or venv environment
python run.py
```

The API server will be available at **http://localhost:8000**.

## API Usage

You can interact with the RAG API using the following endpoints.

### REST API

```bash
curl -X 'POST' \
  'http://localhost:8000/v1/rest-retrieve/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_input": "What is attention mechanism?"
}'
```

### SSE (Streaming) API

```bash
curl -X 'POST' \
  'http://localhost:8000/v1/sse-retrieve/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_input": "What is attention mechanism?"
}'
```

### WebSocket API

1.  Go to the interactive API documentation at **http://localhost:8000/docs**.
2.  Find the `/v1/ws-retrieve/` endpoint and open it.
3.  Click "Try it out" and then "Execute" to establish a connection.
4.  Use the interface to send and receive messages.

## Accessing UIs

- **Airflow UI**: http://localhost:8080
- **FastAPI Docs**: http://localhost:8000/docs
- **Langfuse UI**: http://localhost:3000
- **MinIO Console**: http://localhost:9001
- **LM Studio**: Local application (if using local LLM)

## Troubleshooting

### LM Studio Connection Issues

If you encounter connection errors when using LM Studio with Langfuse:

1. **Ensure LM Studio server is running** on port 1234
2. **Use the correct URL format**: `http://host.docker.internal:1234/v1` for Docker environments
3. **Check LM Studio logs** for any error messages
4. **Verify model is loaded** in LM Studio before testing connections 