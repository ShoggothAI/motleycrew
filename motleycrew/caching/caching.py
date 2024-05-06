import os

from motleycrew.caching.http_cache import (
    BaseHttpCache,
    RequestsHttpCaching,
    HttpxHttpCaching,
    CurlCffiHttpCaching,
)

is_caching = False
caching_http_library_list = [
    RequestsHttpCaching(),
    HttpxHttpCaching(),
    CurlCffiHttpCaching(),
]


def set_strong_cache(val: bool):
    """Enable or disable the strict-caching option"""
    BaseHttpCache.strong_cache = bool(val)


def set_update_cache_if_exists(val: bool):
    """Enable or disable cache updates"""
    BaseHttpCache.update_cache_if_exists = bool(val)


def set_cache_location(location: str) -> str:
    """Set the caching root directory, return the absolute path of the directory"""
    BaseHttpCache.root_cache_dir = location
    return os.path.abspath(BaseHttpCache.root_cache_dir)


def enable_cache():
    """Enable global caching"""
    global is_caching
    for http_cache in caching_http_library_list:
        http_cache.enable()
    is_caching = True


def disable_cache():
    """Disable global caching"""
    global is_caching
    for http_cache in caching_http_library_list:
        http_cache.disable()
    is_caching = False
