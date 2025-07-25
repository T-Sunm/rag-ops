import json
import logging
import asyncio
from functools import wraps
from typing import Any
from uuid import UUID

import redis
from nemoguardrails import LLMRails

from src.config.settings import SETTINGS


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class StandardCache:
    def __init__(self):
        self.storage_uri = f"redis://{SETTINGS.REDIS_URI}"
        self.client = redis.Redis(
            host=self.storage_uri.split(":")[1].replace("//", ""),
            port=int(self.storage_uri.split(":")[2]),
        )

    def _cache_logic(self, func, args, kwargs, ttl, validatedModel, is_async=False):
        """Shared cache logic cho cả sync và async functions"""
        environment = SETTINGS.ENVIRONMENT
        module_name = func.__module__
        func_name = func.__qualname__

        # Bỏ 'self' và LLMRails khỏi args để tránh serialization issues
        if args and hasattr(args[0], func.__name__):
            # Xử lý method: bỏ qua self (args[0]) và lọc LLMRails
            args_to_serialize = tuple(
                arg for arg in args[1:] if not isinstance(arg, LLMRails)
            )
            class_name = args[0].__class__.__name__
            func_name = f"{class_name}.{func.__name__}"
        else:
            # Xử lý function: lọc LLMRails
            args_to_serialize = tuple(
                arg for arg in args if not isinstance(arg, LLMRails)
            )

        # Lọc LLMRails khỏi kwargs
        kwargs_to_serialize = {
            k: v for k, v in kwargs.items() if not isinstance(v, LLMRails)
        }
        # Tạo cache key
        dumped_args = self.serialize(args_to_serialize)
        dumped_kwargs = self.serialize(kwargs_to_serialize)
        key = (
            f"mlops:{environment}:{module_name}:"
            + f"{func_name}:{dumped_args}:{dumped_kwargs}"
        )
        logging.info(f"Cached key: {key}")

        # Kiểm tra Redis connection
        try:
            cached_result = self.client.get(key)
            logging.info(f"Cache lookup result: {cached_result is not None}")
        except Exception as e:
            logging.warning(f"Redis not available for key: {key}, error: {e}")
            return None, None  # Signal: gọi function trực tiếp

        # Cache HIT - trả về kết quả từ cache
        if cached_result:
            logging.info(f"Cache HIT for key: {key}")
            return "hit", self.deserialize(cached_result)

        # Cache MISS - cần gọi function
        logging.info(f"Cache MISS for key: {key}")
        return "miss", key

    def cache(self, *, ttl: int = 60 * 60, validatedModel: Any = None):
        """
        Decorator hỗ trợ cả sync và async functions
        - Tự động detect function type (sync/async)
        - Cache kết quả trong Redis với TTL
        """

        def inner(func):
            is_async = asyncio.iscoroutinefunction(func)

            if is_async:

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    cache_result, data = self._cache_logic(
                        func, args, kwargs, ttl, validatedModel, True
                    )

                    if cache_result is None:  # Redis lỗi
                        return await func(*args, **kwargs)
                    elif cache_result == "hit":  # Cache HIT
                        return data
                    else:  # Cache MISS - gọi function và store kết quả
                        result = await func(*args, **kwargs)
                        self._store_result(data, result, ttl, validatedModel)
                        return result

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    cache_result, data = self._cache_logic(
                        func, args, kwargs, ttl, validatedModel, False
                    )

                    if cache_result is None:  # Redis lỗi
                        return func(*args, **kwargs)
                    elif cache_result == "hit":  # Cache HIT
                        return data
                    else:  # Cache MISS - gọi function và store kết quả
                        result = func(*args, **kwargs)
                        self._store_result(data, result, ttl, validatedModel)
                        return result

                return sync_wrapper

        return inner

    def _store_result(self, key, result, ttl, validatedModel):
        """Lưu kết quả vào cache với validation (nếu có) --- Có 2 trường hợp lưu là 2 kết quả của Langchain và Guardrails"""
        data_to_serialize = None
        # Check if result is a Pydantic model
        if hasattr(result, "model_dump"):
            data_to_serialize = result.model_dump()
        # Check if it's a list of Pydantic models
        elif isinstance(result, list) and result and hasattr(result[0], "model_dump"):
            data_to_serialize = [r.model_dump() for r in result]
        # Otherwise, assume it's a dict or other JSON-serializable type
        else:
            data_to_serialize = result

        try:
            serialized_result = self.serialize(data_to_serialize)
        except TypeError as e:
            logging.warning(f"Could not serialize result for key: {key}, error: {e}")
            return  # Do not cache if serialization fails

        # Validation (optional)
        if validatedModel:
            try:
                validatedModel(**self.deserialize(serialized_result))
            except Exception as e:
                logging.warning(f"Validation failed for key: {key}, error: {e}")
                return  # Không cache nếu validation fail

        # Store vào Redis
        self.set_key(key, serialized_result, ttl)
        logging.info(f"Cache STORED for key: {key}")

    def set_key(self, key: str, value: Any, ttl: int = 60 * 60):
        """Sets key value pair in redis cache"""
        self.client.set(key, value)
        self.client.expire(key, ttl)

    def remove_key(self, key: str):
        """Removes key from redis cache"""
        self.client.delete(key)

    def serialize(self, value: Any) -> str:
        """Serializes the value to json"""
        return json.dumps(value, cls=UUIDEncoder, sort_keys=True)

    def deserialize(self, value: str) -> dict:
        """Deserializes the value from json"""
        return json.loads(value)

    def list_keys(self, pattern: str = f"mlops:{SETTINGS.ENVIRONMENT}:*") -> Any:
        """List all keys in redis cache"""
        return self.client.keys(pattern)


standard_cache = StandardCache()
