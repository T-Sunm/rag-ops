# RAG API Layer - Lab

This project provides a comprehensive API layer for a Retrieval-Augmented Generation (RAG) system. It includes data ingestion pipelines, a FastAPI-based application to serve the RAG model, and an observability stack using Langfuse.

## Architecture Components

This RAG system follows a modular, microservices-oriented architecture with clear separation of concerns:

### 🚀 **API Layer** (`src/`)
The FastAPI-based service layer provides multiple interaction patterns:

| Endpoint | Protocol | Use Case | Port |
|----------|----------|----------|------|
| `/v1/rest-retrieve/` | REST | Standard request/response | 8000 |
| `/v1/sse-retrieve/` | Server-Sent Events | Real-time streaming responses | 8000 |
| `/v1/ws-retrieve/` | WebSocket | Bidirectional real-time communication | 8000 |

### ⚡ **Caching Layer** (`infrastructure/cache/`)
Redis-powered caching system for performance optimization:

- **📊 Embeddings Cache**: Stores computed embeddings (TTL: 24h)
- **🔍 Vector Search Cache**: Caches similarity search results (TTL: 1h)
- **🤖 LLM Response Cache**: Caches model responses (TTL: 6h)
- **👤 Session History Cache**: Maintains conversation context (TTL: 24h)

### 🧠 **RAG Core Engine**
The heart of the retrieval-augmented generation system:

- **🔤 Embeddings Generator**: Converts text to vector representations
- **🗄️ ChromaDB Vector Store**: Persistent vector database (`DATA/chromadb`)
- **🤖 LLM Provider**: OpenAI API or local LM Studio integration
- **🔄 RAG Service**: Orchestrates retrieval and generation pipeline

### 📊 **Data Pipeline** (`ingest_data/`)
Apache Airflow-orchestrated data processing workflow:

| Component | Purpose | Port |
|-----------|---------|------|
| **Apache Airflow** | Workflow orchestration | 8080 |
| **MinIO** | S3-compatible object storage | 9001 |
| **PostgreSQL** | Airflow metadata database | 5432 |
| **Redis** | Airflow message broker | 6379 |

### 📈 **Observability Stack** (`infrastructure/observability/`)
Comprehensive monitoring and tracing with Langfuse:

- **🔍 Request Tracing**: End-to-end request tracking
- **📊 Performance Metrics**: Response times, cache hit rates
- **🤖 LLM Monitoring**: Token usage, model performance
- **🐛 Debug Tools**: Error tracking and troubleshooting

### 🏗️ **Infrastructure Services**
Supporting services for the entire stack:

```
infrastructure/
├── cache/          # Redis cache service
├── observability/  # Langfuse monitoring stack  
└── storage/        # Additional storage services
```

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

This project contains multiple `docker-compose` configurations for different infrastructure components.

#### A. Cache Layer (Redis)

Start the Redis cache service for application-level caching:

```bash
cd infrastructure/cache/
docker compose up -d
```

#### B. Observability (Langfuse)

The Langfuse stack allows you to monitor and debug your RAG application:

```bash
cd infrastructure/observability/
docker compose up -d
```

#### C. Data Ingestion & Storage (Airflow, Minio, etc.)

This stack provides the services needed to run the data ingestion pipeline:

```bash
cd ingest_data/
docker compose up -d
```

### Step 2: Run the Data Ingestion Pipeline

Once the Airflow services are running, you need to trigger the DAG to process your documents.

1.  Navigate to the Airflow UI: **http://localhost:8080**
2.  Login with the default credentials (`airflow`/`airflow`).
3.  Find the DAG named `ingest_pipeline` (or similar) and trigger it manually.
4.  Wait for the DAG to complete successfully. This will process the PDFs in `DATA/data_source` and store the embeddings in ChromaDB.

### Step 3: Start the RAG API Server

After all infrastructure services are up and the data has been ingested, you can start the FastAPI application.

```bash
# Make sure you have activated your conda or venv environment
python run.py --provider groq
```

The API server will be available at **http://localhost:8000**.

> **Note**: The API server now includes Redis caching for improved performance. Make sure the Redis cache service is running before starting the API server.

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

## Accessing UIs

- **FastAPI Docs**: http://localhost:8000/docs
- **Langfuse UI**: http://localhost:3000
- **Airflow UI**: http://localhost:8080
- **MinIO Console**: http://localhost:9001
- **Redis Commander**: http://localhost:8081 (if enabled in cache stack)
- **LM Studio**: Local application (if using local LLM)

## Troubleshooting

### Common Issues

#### Redis Connection Issues

If you encounter Redis connection errors:

1. **Ensure Redis cache service is running**:
   ```bash
   cd infrastructure/cache/
   docker compose ps
   ```
2. **Check Redis logs**:
   ```bash
   docker compose logs redis
   ```
3. **Verify Redis is accessible** on port 6379

#### LM Studio Connection Issues

If you encounter connection errors when using LM Studio with Langfuse:

1. **Ensure LM Studio server is running** on port 1234
2. **Use the correct URL format**: `http://host.docker.internal:1234/v1` for Docker environments
3. **Check LM Studio logs** for any error messages
4. **Verify model is loaded** in LM Studio before testing connections

#### Service Startup Order

Make sure to start services in the correct order:
1. Cache layer (Redis)
2. Observability (Langfuse)
3. Data ingestion (Airflow stack)
4. Run data ingestion pipeline
5. Start RAG API server 