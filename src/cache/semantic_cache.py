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

    def _get_context_str(self, **kwargs: Any) -> Optional[str]:
        """Extracts context string from keyword arguments."""
        question = kwargs.get("question")
        messages = kwargs.get("messages")
        if messages:  # post-cache
            return build_context(messages)
        return question  # pre-cache

    async def _handle_sse_cache_hit(self, hit: Generation):
        """Handles an SSE cache hit by streaming the cached response."""
        try:
            cached_data = json.loads(hit.text)
            response_to_yield = cached_data.get("response", "")
            if response_to_yield and isinstance(response_to_yield, str):
                words = response_to_yield.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield f"{json.dumps(chunk)}\n\n"
                return
        except (json.JSONDecodeError, KeyError):
            yield f'{json.dumps("Error loading from cache")}\n\n'

    async def _execute_and_cache_sse(
        self, func, namespace: str, context_str: str, *args, **kwargs
    ):
        """Executes the function for an SSE cache miss and caches the result."""
        full_response = ""
        async for chunk in func(*args, **kwargs):
            clean_chunk = chunk.replace("\n\n", "")
            try:
                # Try to decode JSON if it looks like JSON (starts and ends with quotes)
                if clean_chunk.strip().startswith('"') and clean_chunk.strip().endswith(
                    '"'
                ):
                    decoded_chunk = json.loads(clean_chunk)
                    full_response += decoded_chunk
                else:
                    full_response += clean_chunk
            except json.JSONDecodeError:
                full_response += clean_chunk  # Append as is if not valid JSON
            yield chunk

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

    def _handle_rest_cache_hit(self, hit: Generation) -> Any:
        """Handles a REST API cache hit."""
        cached_data = json.loads(hit.text)
        return cached_data["response"]

    async def _execute_and_cache_rest(
        self, func, namespace: str, context_str: str, *args, **kwargs
    ):
        """Executes the function for a REST API cache miss and caches the result."""
        result = await func(*args, **kwargs)
        cache_data = {"type": "rest_response", "response": result}
        self._cache.update(
            context_str,
            namespace,
            [Generation(text=json.dumps(cache_data))],
        )
        logger.debug("Cache-miss → stored [%s]: %s", namespace, context_str)
        return result

    def cache(self, *, namespace: str):
        def inner(func):
            if inspect.isasyncgenfunction(func):

                @wraps(func)
                async def sse_wrapper(*args, **kwargs):
                    context_str = self._get_context_str(**kwargs)

                    hits: List[Generation] = self._cache.lookup(context_str, namespace)

                    if hits:
                        logger.info("SSE Cache-hit [%s]: %s", namespace, context_str)
                        async for chunk in self._handle_sse_cache_hit(hits[0]):
                            yield chunk
                    else:
                        async for chunk in self._execute_and_cache_sse(
                            func, namespace, context_str, *args, **kwargs
                        ):
                            yield chunk

                return sse_wrapper
            else:  # Is a coroutine function

                @wraps(func)
                async def rest_wrapper(*args, **kwargs):
                    context_str = self._get_context_str(**kwargs)

                    hits: List[Generation] = self._cache.lookup(context_str, namespace)

                    if hits:
                        logger.info("REST Cache-hit [%s]: %s", namespace, context_str)
                        return self._handle_rest_cache_hit(hits[0])
                    else:
                        return await self._execute_and_cache_rest(
                            func, namespace, context_str, *args, **kwargs
                        )

                return rest_wrapper

        return inner


semantic_cache_llms = SemanticCacheLLMs()
