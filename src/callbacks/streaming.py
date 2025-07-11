from typing import Any
from fastapi import WebSocket
from langchain.callbacks.base import AsyncCallbackHandler


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.websocket.send_text(f"Answer: {token}")
