from typing import Any, Optional
from abc import ABC, abstractmethod
from pprint import pprint


class RetrievableObjectParent(ABC):
    def __init__(self, id: str, name: str, description: Optional[str] = None):
        assert id is not None, "id must be provided"
        assert name is not None, "name must be provided"

        self.id = id
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def summary(self) -> str:
        pass


class SimpleRetrievableObject(RetrievableObjectParent):
    def __init__(
        self, id: str, name: str, payload: Any, description: Optional[str] = None
    ):
        super().__init__(id, name, description)
        self.payload = payload

    @property
    def summary(self) -> str:
        return f"""SimpleRetrievableObject: {self.name}
id: {self.id}
description: {self.description}
payload: {self.payload}"""
