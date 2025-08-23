"""
    A semantic cache for LLM responses that supports both REST API and SSE.

    This class provides a decorator-based caching mechanism that intelligently handles
    two types of function returns:
    1.  **Async Functions (for REST API):** Caches the final, complete string response.
    2.  **Async Generator Functions (for SSE):** Caches both the individual streamed chunks and the full concatenated response.

    The caching strategy is designed for interoperability. When a cache lookup occurs:
    - A REST API call can retrieve a full response that was originally cached from an SSE stream by using the stored `full_response`.
    - An SSE stream can retrieve a response cached by a REST API call and stream it back character-by-character to the client, preserving the streaming experience.

    This prevents compatibility issues where one service type tries to use a cache entry from another, such as a REST API endpoint encountering an array of chunks from an SSE cache.
"""

import inspect
import asyncio
import logging
from functools import wraps
from typing import List, Any, Optional
from langchain_redis import RedisSemanticCache
from src.infrastructure.embeddings.embeddings import embedding_service
from src.utils.text_processing import build_context
from langchain_core.outputs import Generation
import json

logger = logging.getLogger(__name__)


class SemanticCacheLLMs:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6378",
        *,
        embeddings: Optional[Any] = None,
        distance_threshold: float = 0.2,
        ttl: int = 20,
    ):
        self._cache = RedisSemanticCache(
            embeddings=embeddings or embedding_service,
            redis_url=redis_url,
            distance_threshold=distance_threshold,
            ttl=ttl,
            name="llm_cache",
            prefix="llmcache",
        )
        logger.info(
            "SemanticCacheLLMs init (threshold=%s, ttl=%s)",
            distance_threshold,
            ttl,
        )

    def cache(self, *, namespace: str):
        def inner(func):
            is_async_gen = inspect.isasyncgenfunction(func)
            is_async_func = asyncio.iscoroutinefunction(func) and not is_async_gen

            if is_async_gen:  # for sse

                @wraps(func)
                async def wrapper(*args, **kwargs):
                    question = kwargs.get("question")
                    messages = kwargs.get("messages")

                    if messages:  # post-cache
                        context_str = build_context(messages)
                    else:  # pre-cache
                        context_str = question

                    # 1) Lookup
                    hits: List[Generation] = self._cache.lookup(context_str, namespace)
                    if hits:
                        logger.info("SSE Cache-hit [%s]: %s", namespace, context_str)
                        txt = hits[0].text
                        print(txt)

                        try:
                            cached_data = json.loads(txt)
                            response_to_yield = cached_data.get("response", "")

                            if response_to_yield and isinstance(response_to_yield, str):
                                # Stream word by word for smooth UX
                                words = response_to_yield.split(" ")
                                for i, word in enumerate(words):
                                    # Add space back except for last word
                                    chunk = word + (" " if i < len(words) - 1 else "")
                                    yield f"{json.dumps(chunk)}\n\n"

                                return
                        except (json.JSONDecodeError, KeyError):
                            # Fallback for malformed cache
                            yield f"{json.dumps('Error loading from cache')}\n\n"
                            return

                    # 2) Call LLM function
                    full_response = ""
                    async for chunk in func(*args, **kwargs):
                        full_response += chunk
                        yield chunk

                    # 3) Update cache with clean full response
                    cache_data = {
                        "type": "sse_response",
                        "response": full_response.strip(),
                    }
                    self._cache.update(
                        context_str,
                        namespace,
                        [Generation(text=json.dumps(cache_data))],
                    )
                    logger.info("SSE Cache-miss [%s]: %s", namespace, context_str)
                    return

                return wrapper
            elif is_async_func:  # for restAPI

                @wraps(func)
                async def wrapper(*args, **kwargs):
                    question = kwargs.get("question")
                    messages = kwargs.get("messages")

                    if messages:  # post-cache
                        context_str = build_context(messages)
                    else:  # pre-cache
                        context_str = question

                    # 1) Lookup
                    hits: List[Generation] = self._cache.lookup(context_str, namespace)
                    if hits:
                        logger.info("SSE Cache-hit [%s]: %s", namespace, context_str)
                        txt = hits[0].text
                        print(txt)

                        cached_data = json.loads(txt)
                        response_content = cached_data["response"]

                        return response_content

                    # 2) Call LLM function
                    result = await func(*args, **kwargs)

                    # 3) Update cache
                    cache_data = {"type": "rest_response", "response": result}
                    self._cache.update(
                        context_str,
                        namespace,
                        [Generation(text=json.dumps(cache_data))],
                    )
                    logger.debug("Cache-miss → stored [%s]: %s", namespace, context_str)

                    return result

                return wrapper

        return inner


semantic_cache_llms = SemanticCacheLLMs()
