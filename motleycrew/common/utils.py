import sys
from typing import Optional, Sequence
import hashlib
from urllib.parse import urlparse
from langchain_core.messages import BaseMessage

from motleycrew.common.exceptions import ModuleNotInstalledException


def to_str(value: str | BaseMessage | Sequence[str] | Sequence[BaseMessage]) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, BaseMessage):
        return value.content
    else:
        try:
            return "\n".join([to_str(v) for v in value])
        except TypeError:
            raise TypeError(f"Expected str, BaseMessage, or an iterable of them, got {type(value)}")


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


def print_passthrough(x):
    return x


def ensure_module_is_installed(module_name: str, install_command: str = None) -> None:
    """Checking the installation of the module
    Raises:
        ModuleNotInstalledException
    """
    module_path = sys.modules.get(module_name, None)
    if module_path is None:
        raise ModuleNotInstalledException(module_name, install_command)
