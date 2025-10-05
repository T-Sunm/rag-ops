from fastapi import Request
from src.services.application.rag import Rag


def get_rag_service(request: Request) -> Rag:
    return request.app.state.rag_service
