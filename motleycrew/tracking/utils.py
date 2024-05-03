import os
from typing import List, Optional
import logging

from lunary import LunaryCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig, ensure_config

from .callbacks import LlamaIndexLunaryCallbackHandler
from motleycrew.common import LLMFramework


def get_lunary_public_key():
    """Return lunary public key or None"""
    key = (
        os.environ.get("LUNARY_PUBLIC_KEY")
        or os.getenv("LUNARY_APP_ID")
        or os.getenv("LLMONITOR_APP_ID")
    )
    if not key:
        logging.warning("Lunary public key is not set, tracking will be disabled")
    return key


def create_lunary_callback() -> LunaryCallbackHandler:
    """Creation new LunaryCallbackHandler object"""
    lunary_public_key = get_lunary_public_key()
    if lunary_public_key is not None:
        return LunaryCallbackHandler(app_id=lunary_public_key)


def get_llamaindex_default_callbacks():
    """Return tracking for llamaindex platform"""
    _default_callbacks = []

    # init lunary callback
    lunary_public_key = get_lunary_public_key()
    if lunary_public_key is not None:
        _default_callbacks.append(LlamaIndexLunaryCallbackHandler(lunary_public_key))

    return _default_callbacks


def get_langchain_default_callbacks():
    """Return tracking for langchain platform"""
    _default_callbacks = []

    # init lunary callback
    lunary_callback = create_lunary_callback()
    if lunary_callback is not None:
        _default_callbacks.append(lunary_callback)

    return _default_callbacks


DEFAULT_CALLBACKS_MAP = {
    LLMFramework.LANGCHAIN: get_langchain_default_callbacks,
    LLMFramework.LLAMA_INDEX: get_llamaindex_default_callbacks,
}


def get_default_callbacks_list(
    framework_name: str = LLMFramework.LANGCHAIN,
) -> List[BaseCallbackHandler]:
    """Return list of defaults tracking"""
    _default_callbacks = []
    dc_factory = DEFAULT_CALLBACKS_MAP.get(framework_name, None)

    if callable(dc_factory):
        _default_callbacks = dc_factory()
    else:
        msg = "Default callbacks are not implemented for {} framework".format(framework_name)
        logging.warning(msg)

    return _default_callbacks


def combine_callbacks(
    updated_callbacks: List[BaseCallbackHandler],
    updating_callbacks: List[BaseCallbackHandler],
    unique_cls: bool = True,
) -> List[BaseCallbackHandler]:
    """Combining callback lists
    unique_cls: bool - flag adding callback with a unique class
    return : modified updated_callbacks list
    """
    for updating in updating_callbacks:
        if unique_cls and not any(
            isinstance(updating, updated.__class__) for updated in updated_callbacks
        ):
            updated_callbacks.append(updating)
        elif updating not in updating_callbacks:
            updated_callbacks.append(updating)
    return updated_callbacks


def add_default_callbacks_to_langchain_config(
    config: Optional[RunnableConfig] = None,
) -> RunnableConfig:
    """Add default callback to langchain config
    return: modified config
    """
    if config is None:
        config = ensure_config(config)

    _default_callbacks = get_default_callbacks_list()
    if _default_callbacks:
        config_callbacks = config.get("callbacks") or []
        config["callbacks"] = combine_callbacks(config_callbacks, _default_callbacks)
    return config
