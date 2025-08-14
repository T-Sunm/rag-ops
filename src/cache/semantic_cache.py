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
                        cached_chunks = json.loads(txt)
                        for chunk in cached_chunks:
                            yield chunk
                        return
                    # 2) Call LLM function
                    result_chunks = []
                    async for chunk in func(*args, **kwargs):
                        result_chunks.append(chunk)
                        yield chunk

                    # 3) Update cache
                    self._cache.update(
                        context_str,
                        namespace,
                        [Generation(text=json.dumps(result_chunks))],
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
                        logger.debug("Cache-hit [%s]: %s", namespace, context_str)
                        txt = hits[0].text
                        print(txt)
                        return json.loads(txt)
                    # 2) Call LLM function
                    result = await func(*args, **kwargs)

                    # 3) Update cache
                    self._cache.update(
                        context_str, namespace, [Generation(text=json.dumps(result))]
                    )
                    logger.debug("Cache-miss → stored [%s]: %s", namespace, context_str)

                    return result

                return wrapper

        return inner


semantic_cache_llms = SemanticCacheLLMs()
