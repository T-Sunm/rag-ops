from fastapi import APIRouter
from src.routers import rest_retrieval, sse_retrieval, ws_retrieval

api_router = APIRouter()
api_router.include_router(
    rest_retrieval.router, prefix="/rest-retrieve", tags=["REST API Retriever"]
)
api_router.include_router(
    sse_retrieval.router, prefix="/sse-retrieve", tags=["SSE Retriever"]
)
api_router.include_router(
    ws_retrieval.router, prefix="/ws-retrieve", tags=["WebSocket Retriever"]
)
