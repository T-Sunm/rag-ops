from typing import List
from langchain_core.messages import BaseMessage, ToolMessage


def build_context(messages: List[BaseMessage]) -> str:
    tool_chunks = []
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_chunks.append(str(m.content))

    context_str = "\n\n--- Retrieved Documents ---\n\n".join(tool_chunks)
    return context_str


def is_guardrails_error(response) -> bool:
    """Check if response contains guardrails error/blocking"""

    # If response is a dictionary (use for restapi)
    if isinstance(response, dict):
        # Check for direct error key
        if "error" in response:
            return True

    # If response is a string (use for sse)
    response_str = str(response)
    error_indicators = [
        "guardrails_violation",
        "Blocked by self check output rails",
        "content_blocked",
        "I'm sorry, I can't respond to that",
        '"error":',
        "blocked by guardrails",
    ]

    return any(
        indicator.lower() in response_str.lower() for indicator in error_indicators
    )
