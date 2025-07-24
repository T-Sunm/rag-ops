from fastapi import Request
from nemoguardrails import LLMRails


def get_guardrails(request: Request) -> LLMRails:
    return request.app.state.rails
