import pytest

from motleycrew.caсhing.http_cache import RequestsHttpCaching, CacheException
from motleycrew.caсhing import http_cache


@pytest.fixture
def requests_cash():
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
def test_white_list(requests_cash, url, expected_result):
    http_cache.CACHE_WHITELIST = ["*//api.openai.com/v1/*"]
    http_cache.CACHE_BLACKLIST = []
    assert requests_cash.should_cache(url) == expected_result


@pytest.mark.parametrize(
    "url, expected_result",
    [
        ("https://api.lunary.ai/v1/runs/ingest", False),
        ("https://api.openai.com/v1/chat/completions", True),
        ("https://duckduckgo.com/", True),
        ("https://links.duckduckgo.com/d.j", False),
    ],
)
def test_black_list(requests_cash, url, expected_result):
    http_cache.CACHE_WHITELIST = []
    http_cache.CACHE_BLACKLIST = ["*//api.lunary.ai/v1/*", "*//links.duckduckgo.com/*"]
    assert requests_cash.should_cache(url) == expected_result


def test_raise_cache_lists(requests_cash):
    http_cache.CACHE_WHITELIST = ["*//links.duckduckgo.com/*"]
    http_cache.CACHE_BLACKLIST = ["*//api.lunary.ai/v1/*"]
    url = "https://api.openai.com/v1/chat/completions"
    with pytest.raises(CacheException):
        requests_cash.should_cache(url)
