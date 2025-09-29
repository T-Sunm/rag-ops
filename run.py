import os
import argparse

parser = argparse.ArgumentParser(description="Run RAG API server")
parser.add_argument(
    "--provider",
    choices=["groq", "openai", "lm-studio"],
    help="LLM provider to use (e.g., groq, openai, lm-studio)",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="environment_battery",
    help="Dataset name to use for RAG (must match ingested dataset name)",
)
# parser.add_argument(
#     "--host",
#     type=str,
#     default="0.0.0.0",
#     help="Host to bind the server to",
# )
# parser.add_argument(
#     "--port",
#     type=int,
#     default=8000,
#     help="Port to bind the server to",
# )

# Use parse_known_args() to be compatible with uvicorn's reloader,
# which might add its own arguments.
args, _ = parser.parse_known_args()

# Set environment variables
os.environ["LITELLM_MODEL"] = args.provider
os.environ["DATASET_NAME"] = args.dataset  # Set the dataset for RAG service


import uvicorn
from src.config.settings import SETTINGS
from dotenv import load_dotenv

load_dotenv()


def main():
    print(f"🚀 Starting RAG API server...")
    print(f"   Provider: {args.provider}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Collection: rag-pipeline-{args.dataset}")
    print(f"   Host: {SETTINGS.HOST}:{SETTINGS.PORT}")
    print(f"   API Docs: http://{SETTINGS.HOST}:{SETTINGS.PORT}/docs")

    uvicorn_config = {
        "app": "src.main:app",
        "host": SETTINGS.HOST,
        "port": SETTINGS.PORT,
        "reload": True,
    }

    # Start Uvicorn server
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
