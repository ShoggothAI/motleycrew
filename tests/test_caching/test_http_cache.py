import os
from typing import Callable, Any
import asyncio

import pytest
import requests
import httpx
import curl_cffi

from motleycrew.caching.http_cache import (
    BaseHttpCache,
    RequestsHttpCaching,
    HttpxHttpCaching,
    CurlCffiHttpCaching,
    CacheException,
    StrongCacheException,
)
from motleycrew.caching import (
    set_cache_whitelist,
    set_cache_blacklist,
    set_cache_location,
    enable_cache,
    disable_cache,
    set_strong_cache,
)

ROOT_TEST_CACHE_LOCATION = "tests/test_caching/cache"
strong_cache_url = "https://google.com"
test_url = "https://example.com"


def get_library_method(library: str) -> Callable:
    if library == "requests":
        return requests.api.request
    elif library == "httpx":
        return httpx.Client.send
    elif library == "curl_cffi":
        from curl_cffi.requests import AsyncSession
        return AsyncSession.request


def get_caching_obj(library: str) -> BaseHttpCache:
    if library == "requests":
        return RequestsHttpCaching()
    elif library == "httpx":
        return HttpxHttpCaching()
    elif library == "curl_cffi":
        return CurlCffiHttpCaching()


def make_request(library: str, url: str = test_url) -> Any:
    if library == "requests":
        return requests.get(url)
    elif library == "httpx":
        client = httpx.Client()
        return client.get(url)
    elif library == "curl_cffi":
        from curl_cffi.requests import AsyncSession
        session = AsyncSession()
        return asyncio.run(session.request("GET", url))


@pytest.fixture
def requests_cache():
    return RequestsHttpCaching()


@pytest.mark.parametrize(
    "url, expected_result",
    [
        ("https://api.lunary.ai/v1/runs/ingest", False),
        ("https://api.openai.com/v1/chat/completions", True),
        ("https://duckduckgo.com/", False),
        ("https://links.duckduckgo.com/d.j", False),
    ],
)
def test_cache_whitelist(requests_cache, url, expected_result):
    set_cache_whitelist(["*//api.openai.com/v1/*"])
    assert requests_cache.should_cache(url) == expected_result


@pytest.mark.parametrize(
    "url, expected_result",
    [
        ("https://api.lunary.ai/v1/runs/ingest", False),
        ("https://api.openai.com/v1/chat/completions", True),
        ("https://duckduckgo.com/", True),
        ("https://links.duckduckgo.com/d.j", False),
    ],
)
def test_cache_blacklist(requests_cache, url, expected_result):
    set_cache_blacklist(["*//api.lunary.ai/v1/*", "*//links.duckduckgo.com/*"])
    assert requests_cache.should_cache(url) == expected_result


def test_exception_if_both_lists_set(requests_cache):
    requests_cache.cache_whitelist = ["*//links.duckduckgo.com/*"]
    requests_cache.cache_blacklist = ["*//api.lunary.ai/v1/*"]
    url = "https://api.openai.com/v1/chat/completions"
    with pytest.raises(CacheException):
        requests_cache.should_cache(url)


@pytest.mark.parametrize("library", [requests, httpx, curl_cffi])
def test_http_cache_objects(library):
    library_name = library.__name__
    test_cache_location = os.path.join(ROOT_TEST_CACHE_LOCATION, library_name)
    set_cache_location(test_cache_location)
    caching_obj = get_caching_obj(library_name)
    library_method = get_library_method(library_name)

    enable_cache()

    assert library_method == caching_obj.library_method
    assert get_library_method(library_name) != library_method

    set_strong_cache(True)
    with pytest.raises(StrongCacheException):
        make_request(library_name, strong_cache_url)
    try:
        response = make_request(library_name)
    except Exception:
        response = None
    assert response

    set_strong_cache(False)
    disable_cache()
    assert library_method == get_library_method(library_name)
