from fastapi import Request
from nemoguardrails import LLMRails


def get_guardrails_restapi(request: Request) -> LLMRails:
    return request.app.state.rails_restapi


def get_guardrails_sse(request: Request) -> LLMRails:
    return request.app.state.rails_sse
