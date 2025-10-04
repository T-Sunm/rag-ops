import logging
import tracemalloc
import os
import argparse
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv

load_dotenv()

from src.api.routers.api import api_router
from src.services.application.rag import rag_service
from src.config.settings import APP_CONFIGS, SETTINGS


tracemalloc.start()


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return (
            record.args is not None
            and len(record.args) >= 3
            and list(record.args)[2] not in ["/health", "/ready"]
        )


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = rag_service

    config_restapi = RailsConfig.from_path(SETTINGS.GUARDRAILS_RESTAPI_PATH)
    app.state.rails_restapi = LLMRails(config_restapi)

    config_sse = RailsConfig.from_path(SETTINGS.GUARDRAILS_SSE_PATH)
    app.state.rails_sse = LLMRails(config_sse)

    yield


app = FastAPI(**APP_CONFIGS, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
async def readycheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(
    api_router,
    prefix=SETTINGS.API_V1_STR,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG API server")
    parser.add_argument(
        "--provider",
        choices=["groq", "openai", "lm-studio"],
        required=True,
        help="LLM provider to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="environment_battery",
        help="Dataset name to use for RAG",
    )
    args = parser.parse_args()

    os.environ["LITELLM_MODEL"] = args.provider
    os.environ["DATASET_NAME"] = args.dataset

    print(f"🚀 Starting RAG API server...")
    print(f"   Provider: {args.provider}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Collection: rag-pipeline-{args.dataset}")
    print(f"   Host: {SETTINGS.HOST}:{SETTINGS.PORT}")
    print(f"   API Docs: http://{SETTINGS.HOST}:{SETTINGS.PORT}/docs")

    uvicorn.run(
        "src.main:app",
        host=SETTINGS.HOST,
        port=SETTINGS.PORT,
        reload=True,
    )
