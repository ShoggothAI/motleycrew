from motleycrew.caсhing.http_cache import (
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
