import os

from motleycrew.caсhing.http_cache import (
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
    """Enabling disabling the strictly caching option"""
    BaseHttpCache.strong_cache = bool(val)


def set_cache_location(location: str) -> str:
    """Sets the caching root directory, returns the absolute path of the derrictory"""
    BaseHttpCache.root_cache_dir = location
    return os.path.abspath(BaseHttpCache.root_cache_dir)


def enable_cache():
    """The function of enable the caching process"""
    global is_caching
    for http_cache in caching_http_library_list:
        http_cache.enable()
    is_caching = True


def disable_cache():
    """The function of disable the caching process"""
    global is_caching
    for http_cache in caching_http_library_list:
        http_cache.disable()
    is_caching = False
