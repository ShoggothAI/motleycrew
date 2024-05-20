import pytest

from motleycrew.caching.http_cache import RequestsHttpCaching, CacheException
from motleycrew.caching import set_cache_whitelist, set_cache_blacklist


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
