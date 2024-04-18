from typing import Sequence
import logging
from langchain_core.messages import BaseMessage


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


def configure_logging(verbose: bool = False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=level)
