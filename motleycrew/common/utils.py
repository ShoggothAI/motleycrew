from typing import Sequence
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
