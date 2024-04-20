from typing import Optional, Sequence
import logging
import hashlib
from urllib.parse import urlparse
from langchain_core.messages import BaseMessage


def configure_logging(verbose: bool = False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=level)


def to_str(value: str | BaseMessage | Sequence[str] | Sequence[BaseMessage]) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, BaseMessage):
        return value.content
    else:
        try:
            return "\n".join([to_str(v) for v in value])
        except TypeError:
            raise TypeError(
                f"Expected str, BaseMessage, or an iterable of them, got {type(value)}"
            )


def is_http_url(url):
    try:
        parsed_url = urlparse(url)
        return parsed_url.scheme in ["http", "https"]
    except ValueError:
        return False


def generate_hex_hash(data: str, length: Optional[int] = None):
    hash_obj = hashlib.sha256()
    hash_obj.update(data.encode("utf-8"))
    hex_hash = hash_obj.hexdigest()

    if length is not None:
        hex_hash = hex_hash[:length]
    return hex_hash
