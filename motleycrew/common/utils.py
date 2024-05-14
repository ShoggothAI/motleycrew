import sys
import inspect
from typing import Optional, Sequence, List, Union
import logging
import hashlib
from urllib.parse import urlparse
from langchain_core.messages import BaseMessage
from collections import namedtuple

from motleycrew.common.exceptions import NotInstallModuleException

InstallCheckedModule = namedtuple(
    "InstallCheckedModule",
    ["module_name", "attr_name", "install_command"],
    defaults=[None, None],
)


def configure_logging(verbose: bool = False, debug: bool = False):
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
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


def check_install_module(
    module: Union[InstallCheckedModule, List[InstallCheckedModule]]
) -> None:
    """Checking the installation of the module and the availability of the necessary argument
    Raises:
        NotInstallModuleException
    """
    if isinstance(module, InstallCheckedModule):
        modules = [module]
    else:
        modules = module

    for _module in modules:
        module_path = sys.modules.get(_module.module_name, None)
        if module_path is None:
            raise NotInstallModuleException(
                _module.module_name, install_command=_module.install_command
            )

        if _module.attr_name is not None:
            attrs_names = [
                attr_data[0] for attr_data in inspect.getmembers(_module.module_name)
            ]
            if _module.attr_name not in attrs_names:
                raise NotInstallModuleException(
                    _module.module_name, _module.attr_name, _module.install_command
                )
