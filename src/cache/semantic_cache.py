from langchain_redis import RedisSemanticCache
from langchain.globals import set_llm_cache, get_llm_cache
from src.infrastructure.embeddings.embeddings import embedding_service
from src.config.settings import SETTINGS
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self):
        self.caches = {}
        self.redis_url = f"redis://{SETTINGS.REDIS_URI}"
    
    def get_cache(self, 
                  namespace: str, 
                  distance_threshold: float = 0.2, 
                  ttl: int = 3600):
        """Get or create cache for specific namespace"""
        if namespace not in self.caches:
            try:
                self.caches[namespace] = RedisSemanticCache(
                    embeddings=embedding_service,
                    redis_url=self.redis_url,
                    distance_threshold=distance_threshold,
                    ttl=ttl,
                    name=f"{namespace}_cache",        
                    prefix=f"llmcache:{namespace}"    
                )
                logger.info(f"Created semantic cache: {namespace} (threshold={distance_threshold}, ttl={ttl})")
            except Exception as e:
                logger.error(f"Failed to create cache for {namespace}: {e}")
                return None
        
        return self.caches[namespace]
    
    # 
    def cache(self, 
              namespace: str, 
              distance_threshold: float = 0.2, 
              ttl: int = 3600):
        """
        Decorator for semantic caching with isolated Redis databases/indexes.
        Purpose: Each namespace creates separate Redis storage to prevent cache collision
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Backup current cache
                original_cache = get_llm_cache()
                
                # Get/create specific cache for this namespace
                cache = self.get_cache(namespace, distance_threshold, ttl)
                if cache:
                    set_llm_cache(cache)
                    logger.debug(f"Using semantic cache: {namespace}")
                else:
                    logger.warning(f"Failed to set cache for {namespace}, using original")
                
                try:
                    # ✅ CRITICAL: Call function and return result
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    # Restore original cache
                    if original_cache:
                        set_llm_cache(original_cache)
                    else:
                        set_llm_cache(None)
                        
            return wrapper
        return decorator
    
    def clear_cache(self, namespace: str):
        """Clear cache for specific namespace or all"""
        if namespace:
            if namespace in self.caches:
                self.caches[namespace].clear()
                logger.info(f"Cleared cache for namespace: {namespace}")
        else:
            for name, cache in self.caches.items():
                cache.clear()
                logger.info(f"Cleared cache: {name}")
    
    def get_cache_info(self):
        """Debug info about all caches"""
        info = {}
        for name, cache in self.caches.items():
            info[name] = {
                "index_name": cache.name(),
                "distance_threshold": cache.distance_threshold,
                "ttl": cache.ttl
            }
        return info

# Global instance để dùng như decorator
semantic_cache = SemanticCache()