# Data Ingestion Pipeline

This directory contains the automated data ingestion pipeline, orchestrated by Apache Airflow. It is designed to be easily customizable, allowing you to ingest documents from various sources for your own RAG (Retrieval-Augmented Generation) topics.

The pipeline automates the entire ETL process for your documents:
1.  **Download**: Fetches source documents (e.g., PDFs from URLs).
2.  **Chunk**: Splits documents into smaller, semantically meaningful chunks.
3.  **Embed**: Converts each chunk into a vector embedding.
4.  **Store**: Saves the embeddings into a dedicated collection in ChromaDB.

## How to Ingest Your Own Custom Data

Follow these steps to configure the pipeline for your own topics and data sources.

### Step 1: Add Your Data Sources

The core of the customization happens in one file: `plugins/jobs/download.py`.

1.  Open `ingest_data/plugins/jobs/download.py`.
2.  You will see a dictionary named `DATASETS`. This is where you define your data topics. Each key in the dictionary represents a unique dataset.
3.  You can either **modify an existing dataset** or **add a new one**.

**Example: Adding a new dataset for "finance_reports"**

Add a new entry to the `DATASETS` dictionary. Follow the existing structure:

```python
# ingest_data/plugins/jobs/download.py

DATASETS = {
    "environment_battery": {
        # ... existing data ...
    },
    "llm_papers": {
        # ... existing data ...
    },
    # Add your new dataset here
    "finance_reports": {
        "data": [
            {
                "title": "Annual Financial Report 2023",
                "url": "https://example.com/reports/annual_2023.pdf",
            },
            {
                "title": "Q4 Earnings Call Transcript",
                "url": "https://example.com/reports/q4_earnings.pdf",
            },
        ],
        "description": "Financial reports and earnings transcripts."
    }
}
```

The system is fully automated based on this configuration. You do not need to manually create folders or database collections.

### Step 2: Run the Pipeline

Once you have added your dataset, the system will automatically handle the rest.

1.  **Start the infrastructure**: If it's not already running, start the Airflow services from the `ingest_data` directory:
    ```bash
    docker compose up -d
    ```

2.  **Access the Airflow UI**: Open your browser and navigate to `http://localhost:8080`. Log in with the default credentials (`airflow`/`airflow`).

3.  **Trigger Your DAG**: On the Airflow dashboard, you will find a new DAG automatically generated for your dataset. The DAG will be named based on the key you added in `download.py`. For our example, it would be `ingest_data_finance_reports`.

    Find your DAG, un-pause it (toggle on the left), and click the "play" button on the right to trigger a manual run.

### Step 3: Verify the Ingestion

After the DAG run completes successfully, your data will be ready for the RAG service. You can verify the process:

-   **Downloaded Files**: Check the `infrastructure/storage/data_source/` directory. A new subfolder named after your dataset (e.g., `finance_reports`) will be created, containing the downloaded PDFs.
-   **Vector Embeddings**: The pipeline will also create a new, unique collection in ChromaDB for your data (e.g., `rag-pipeline-finance_reports`). The vector store is persisted in `infrastructure/storage/chromadb/`.

That's it! Your custom data is now ingested and ready to be used by the main application.
