from typing import Any

from pydantic import BaseModel


class ObjectRetrievalMetadata(BaseModel):
    id: str
    name: str
    description: str


class ObjectRetrievalResult(BaseModel):
    metadata: ObjectRetrievalMetadata
    payload: Any
