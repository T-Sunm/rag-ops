import json
import logging
from functools import wraps
from typing import Any
from uuid import UUID

import redis

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
    
    def cache(
        self, *, ttl: int = 60 * 60, validatedModel: Any = None
    ):  # 1 hour
        """Caches decorator for caching function results

        Args:
            func: function to cache
            ttl (int, optional): Time-to-live a.k.a
                The time in seconds to cache the return value.
                Defaults to 60 * 60 seconds a.k.a 1 hour.
            validatedModel (Any, optional): Pydantic model
                to validate the response.
                If validated, result of api will be cached.
                If not validated, result of api will not be cached.
                Defaults to None.
        """

        def inner(func):
            @wraps(func)
            async def inner_cache(*args, **kwargs):
                environment = SETTINGS.ENVIRONMENT
                module_name = func.__module__
                func_name = func.__qualname__
                
                # For method calls, exclude 'self' from args to avoid serialization issues
                # Check if object has the function
                if args and hasattr(args[0], func.__name__):
                    # This is a method call, exclude 'self' from serialization
                    args_to_serialize = args[1:]
                    # Add class name to make cache key more specific
                    class_name = args[0].__class__.__name__
                    func_name = f"{class_name}.{func.__name__}"
                else:
                    # This is a function call, include all args
                    args_to_serialize = args
                
                dumped_args = self.serialize(args_to_serialize)
                dumped_kwargs = self.serialize(kwargs)
                key = (
                    f"mlops:{environment}:{module_name}:"
                    + f"{func_name}:{dumped_args}:{dumped_kwargs}"
                )
                logging.info(f"Cached key: {key}")

                try:
                    cached_result = self.client.get(key)
                    logging.info(f"Cache lookup result: {cached_result is not None}")
                except Exception as e:
                    # If redis is not available, return the actual result
                    logging.warning(
                        f"Redis is not available. "
                        f"Calling the function directly for key: {key}"
                        f" with error: {e}"
                    )
                    return await func(*args, **kwargs)

                # If key is not found in cache,
                # call the function and cache the result.
                if not cached_result:
                    logging.info(f"Cache MISS for key: {key}")  
                    result = await func(*args, **kwargs)
                    try:
                        # If type of result do not contains any custom class
                        serialized_result = self.serialize(result)
                    except TypeError:
                        # If a TypeError is raised, try to convert the result
                        # to a dictionary before serializing
                        if isinstance(result, list):
                            serialized_result = self.serialize(
                                [r.model_dump() for r in result]
                            )
                        else:
                            serialized_result = self.serialize(
                                result.model_dump()
                            )

                    if validatedModel:
                        try:
                            # Validate the result
                            validatedModel(
                                **self.deserialize(serialized_result)
                            )
                        except Exception:
                            # If validation fails, log the error
                            # DO NOT cache and return the actual result
                            logging.warning(
                                f"Validation failed for key: {key} "
                                "with error: {e}"
                            )
                            return self.deserialize(serialized_result)

                    self.set_key(key, serialized_result, ttl)
                    logging.info(f"Cache STORED for key: {key}")
                    return self.deserialize(serialized_result)
                else:
                    # If key is found in cache,
                    # return the cached result after deserializing
                    logging.info(f"Cache HIT for key: {key}")
                    return self.deserialize(cached_result)  # type: ignore

            return inner_cache

        return inner

    def set_key(self, key: str, value: Any, ttl: int = 60 * 60):
        """Sets key value pair in redis cache

        Args:
            key (str): key to set in redis cache
            value (Any): value to set in redis cache
            ttl (int, optional): Time-to-live a.k.a
                The time in seconds to cache the return value.
                Defaults to 60 * 60 = 3600 seconds a.k.a 1 hours.
        """
        self.client.set(key, value)
        self.client.expire(key, ttl)

    def remove_key(self, key: str):
        """Removes key from redis cache

        Args:
            key (str): key to remove in redis cache
        """
        self.client.delete(key)

    def serialize(self, value: Any) -> str:
        """Serializes the value to json

        Args:
            value (Any): value to serialize

        Returns:
            str: serialized value (json string)
        """
        return json.dumps(value, cls=UUIDEncoder, sort_keys=True)

    def deserialize(self, value: str) -> dict:
        """Deserializes the value from json

        Args:
            value (str): json string to deserialize

        Returns:
            json: deserialized value
        """
        return json.loads(value)

    def list_keys(
        self, pattern: str = f"mlops:{SETTINGS.ENVIRONMENT}:*"
    ) -> Any:
        """List all keys in redis cache

        Args:
            pattern (str): pattern to search in keys

        Returns:
            list: list of keys
        """
        return self.client.keys(pattern)


standard_cache = StandardCache()
