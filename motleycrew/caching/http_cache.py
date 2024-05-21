import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Callable, Any, Union
from urllib.parse import urlparse
import logging
import inspect
import fnmatch
import cloudpickle
import platformdirs
import traceback

import requests
from requests.structures import CaseInsensitiveDict
from httpx import (
    Client as HTTPX__Client,
    AsyncClient as HTTPX_AsyncClient,
    Headers as HTTPX__Headers,
)
from curl_cffi.requests import AsyncSession as CurlCFFI__AsyncSession
from curl_cffi.requests import Headers as CurlCFFI__Headers


try:
    from lunary import track_event, run_ctx, event_queue_ctx

    do_update_lunary_event = True
except ImportError:
    track_event = None
    run_ctx = None
    event_queue_ctx = None
    do_update_lunary_event = False


from motleycrew.common.enums import LunaryEventName, LunaryRunType
from .utils import recursive_hash, shorten_filename, FakeRLock


FORCED_CACHE_BLACKLIST = [
    "*//api.lunary.ai/*",
]

CACHE_FILENAME_LENGTH_LIMIT = 64


class CacheException(Exception):
    """Exception for caching process"""


class StrongCacheException(BaseException):
    """Exception use of cache only"""


def file_cache(http_cache: "BaseHttpCache", updating_parameters: dict = {}):
    """Decorator to cache function output based on its inputs, ignoring specified parameters."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            kwargs.update(updating_parameters)
            return http_cache.get_response(func, *args, **kwargs)

        return wrapper

    return decorator


def afile_cache(http_cache: "BaseHttpCache", updating_parameters: dict = {}):
    """Async decorator to cache function output based on its inputs, ignoring specified parameters."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            kwargs.update(updating_parameters)
            return await http_cache.aget_response(func, *args, **kwargs)

        return wrapper

    return decorator


class BaseHttpCache(ABC):
    """Basic abstract class for replacing http library methods"""

    ignore_params: List[str] = []  # ignore params names for create hash file name
    library_name: str = ""
    app_name = os.environ.get("APP_NAME") or "motleycrew"
    root_cache_dir = platformdirs.user_cache_dir(app_name)
    strong_cache: bool = False
    update_cache_if_exists: bool = False
    cache_blacklist: List[str] = []
    cache_whitelist: List[str] = []

    def __init__(self, *args, **kwargs):
        self.is_caching = False

    @abstractmethod
    def get_url(self, *args, **kwargs) -> str:
        """Finds the url in the arguments and returns it"""

    @abstractmethod
    def _enable(self):
        """Replacing the original function with a caching function"""

    @abstractmethod
    def _disable(self):
        """Replacing the caching function with the original one"""

    def enable(self):
        """Enable caching"""
        self._enable()
        self.is_caching = True

        library_log = "for {} library.".format(self.library_name) if self.library_name else "."
        logging.info("Enable caching {} class {}".format(self.__class__, library_log))

    def disable(self):
        """Disable caching"""
        self._disable()
        self.is_caching = False

        library_log = "for {} library.".format(self.library_name) if self.library_name else "."
        logging.info("Disable caching {} class {}".format(self.__class__, library_log))

    def prepare_response(self, response: Any) -> Any:
        """Preparing the response object before saving"""
        return response

    def should_cache(self, url: str) -> bool:
        if self.match_url(url, FORCED_CACHE_BLACKLIST):
            return False

        if self.cache_whitelist and self.cache_blacklist:
            raise CacheException(
                "You can't use both cache whitelist and blacklist at the same time."
            )
        elif self.cache_whitelist:
            return self.match_url(url, self.cache_whitelist)
        elif self.cache_blacklist:
            return not self.match_url(url, self.cache_blacklist)
        return True

    def get_cache_file(self, func: Callable, *args, **kwargs) -> Union[tuple, None]:
        url = self.get_url(*args, **kwargs)
        url_parsed = urlparse(url)

        # Check valid url
        if not self.should_cache(url):
            logging.info("Ignore url to cache: {}".format(url))
            return None

        # check or create cache dirs
        root_dir = Path(self.root_cache_dir)

        cache_dir = (
            root_dir
            / shorten_filename(url_parsed.hostname, length=CACHE_FILENAME_LENGTH_LIMIT)
            / shorten_filename(
                url_parsed.path.strip("/").replace("/", "_"), length=CACHE_FILENAME_LENGTH_LIMIT
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Convert args to a dictionary based on the function's signature
        args_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        args_dict = dict(zip(args_names, args))

        # Remove ignored params
        kwargs_clone = kwargs.copy()
        for param in self.ignore_params:
            args_dict.pop(param, None)
            kwargs_clone.pop(param, None)

        # Create hash based on argument names, argument values, and function source code
        hashing_base = [args_dict, kwargs_clone, inspect.getsource(func)]
        call_hash = shorten_filename(
            recursive_hash(hashing_base, ignore_params=self.ignore_params),
            length=CACHE_FILENAME_LENGTH_LIMIT,
        )

        cache_file = cache_dir / "{}.pkl".format(call_hash)
        return cache_file, url

    def get_response(self, func: Callable, *args, **kwargs) -> Any:
        """Returns a response from the cache if it is found, or executes the request first"""
        cache_data = self.get_cache_file(func, *args, **kwargs)
        if cache_data is None:
            return func(*args, **kwargs)
        cache_file, url = cache_data

        # If cache exists, load and return it
        result = self.load_cache_response(cache_file, url)
        if result is not None:
            if do_update_lunary_event:
                self._update_lunary_event(run_ctx.get())
            return result

        # Otherwise, call the function and save its result to the cache
        result = func(*args, **kwargs)

        self.write_to_cache(result, cache_file, url)
        return result

    async def aget_response(self, func: Callable, *args, **kwargs) -> Any:
        """Async returns a response from the cache if it is found, or executes the request first"""
        cache_data = self.get_cache_file(func, *args, **kwargs)
        if cache_data is None:
            return await func(*args, **kwargs)
        cache_file, url = cache_data

        #  If cache exists, load and return it
        result = self.load_cache_response(cache_file, url)
        if result is not None:
            if do_update_lunary_event:
                self._update_lunary_event(run_ctx.get())
            return result

        # Otherwise, call the function and save its result to the cache
        result = await func(*args, **kwargs)

        self.write_to_cache(result, cache_file, url)
        return result

    @staticmethod
    def _update_lunary_event(
        run_id: str, run_type: str = LunaryRunType.LLM, is_cache: bool = True
    ) -> None:
        """Updating lunary event"""

        if not do_update_lunary_event:
            return

        event_params = {
            "run_type": run_type,
            "event_name": LunaryEventName.UPDATE,
            "run_id": run_id,
            "callback_queue": event_queue_ctx.get(),
        }
        if is_cache:
            event_params["metadata"] = {"cache": True}

        try:
            track_event(**event_params)
        except Exception as exc:
            msg = "[Lunary] An error occurred while updating lunary event {}: {}\n{}".format(
                run_id, exc, traceback.format_exc()
            )
            logging.warning(msg)
            raise exc

    def load_cache_response(self, cache_file: Path, url: str) -> Union[Any, None]:
        """Loads and returns the cached response"""
        if cache_file.exists() and not self.update_cache_if_exists:
            return self.read_from_cache(cache_file, url)
        elif self.strong_cache:
            msg = "Cache file not found: {}\nthe strictly caching option is enabled.".format(
                str(cache_file)
            )
            raise StrongCacheException(msg)

    def read_from_cache(self, cache_file: Path, url: str = "") -> Union[Any, None]:
        """Reads and returns a serialized object from a file"""
        try:
            with cache_file.open("rb") as f:
                logging.info("Used cache for {} url from {}".format(url, cache_file))
                result = cloudpickle.load(f)
                return result
        except Exception as e:
            logging.warning("Unpickling failed for {}".format(cache_file))
            if self.strong_cache:
                msg = "Error reading cached file: {}\n{}".format(str(e), str(cache_file))
                raise StrongCacheException(msg)
        return None

    def write_to_cache(self, response: Any, cache_file: Path, url: str = "") -> None:
        """Writes the response object to a file"""
        response = self.prepare_response(response)
        try:
            with cache_file.open("wb") as f:
                cloudpickle.dump(response, f)
                logging.info("Write cache for {} url to {}".format(url, cache_file))
        except Exception as e:
            logging.warning("Pickling failed for {} url: {}".format(cache_file, e))

    @staticmethod
    def match_url(url: str, patterns: List[str]) -> bool:
        """Checking the url for a match in the list of templates"""
        return any([fnmatch.fnmatch(url, pat) for pat in patterns])


class RequestsHttpCaching(BaseHttpCache):
    """Requests library caching"""

    ignore_params = ["timestamp", "runId", "parentRunId"]
    library_name = "requests"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library_method = requests.api.request

    def get_url(self, *args, **kwargs) -> str:
        """Finds the url in the arguments and returns it"""
        return args[1]

    def prepare_response(self, response: Any) -> Any:
        """Preparing the response object before saving"""
        response.headers = CaseInsensitiveDict()
        response.request.headers = CaseInsensitiveDict()
        return response

    def _enable(self):
        """Replacing the original function with a caching function"""

        @file_cache(self)
        def request_func(*args, **kwargs):
            return self.library_method(*args, **kwargs)

        requests.api.request = request_func

    def _disable(self):
        """Replacing the caching function with the original one"""
        requests.api.request = self.library_method


class HttpxHttpCaching(BaseHttpCache):
    """Httpx library caching"""

    ignore_params = ["s", "headers", "stream", "extensions"]
    library_name = "Httpx"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library_method = HTTPX__Client.send
        self.alibrary_method = HTTPX_AsyncClient.send

    def get_url(self, *args, **kwargs) -> str:
        """Finds the url in the arguments and returns it"""
        return str(args[1].url)

    def prepare_response(self, response: Any) -> Any:
        """Preparing the response object before saving"""
        response.headers = HTTPX__Headers()
        response.request.headers = HTTPX__Headers()
        return response

    def _enable(self):
        """Replacing the original function with a caching function"""

        @file_cache(self, updating_parameters={"stream": False})
        def request_func(s, request, *args, **kwargs):
            return self.library_method(s, request, **kwargs)

        @afile_cache(self, updating_parameters={"stream": False})
        async def arequest_func(s, request, *args, **kwargs):
            return await self.alibrary_method(s, request, **kwargs)

        HTTPX__Client.send = request_func
        HTTPX_AsyncClient.send = arequest_func

    def _disable(self):
        """Replacing the caching function with the original one"""
        HTTPX__Client.send = self.library_method
        HTTPX_AsyncClient.send = self.alibrary_method


class CurlCffiHttpCaching(BaseHttpCache):
    """Curl Cffi library caching"""

    ignore_params = ["s"]
    library_name = "Curl cffi"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library_method = CurlCFFI__AsyncSession.request

    def get_url(self, *args, **kwargs) -> str:
        """Finds the url in the arguments and returns it"""
        return args[2]

    def prepare_response(self, response: Any) -> Any:
        """Preparing the response object before saving"""
        response.headers = CurlCFFI__Headers()
        response.request.headers = CurlCFFI__Headers()
        response.curl = None
        response.cookies.jar._cookies_lock = FakeRLock()
        return response

    def _enable(self):
        """Replacing the original function with a caching function"""

        @afile_cache(self)
        async def request_func(s, method, url, *args, **kwargs):
            return await self.library_method(s, method, url, *args, **kwargs)

        CurlCFFI__AsyncSession.request = request_func

    def _disable(self):
        """Replacing the caching function with the original one"""
        CurlCFFI__AsyncSession.request = self.library_method
