from typing import Optional, Any
from pydantic import BaseModel


class MotleyGraphNode(BaseModel):
    __label__: Optional[str] = None

    @property
    def id(self) -> Optional[Any]:
        return getattr(self, "_id", None)

    @classmethod
    def get_label(cls) -> str:
        if cls.__label__:
            return cls.__label__
        return cls.__name__
