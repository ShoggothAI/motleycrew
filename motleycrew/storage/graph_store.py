from abc import ABC, abstractmethod
from typing import Optional, Any


class MotleyGraphStore(ABC):
    @abstractmethod
    def check_entity_exists(self, entity_id: int) -> bool:
        pass

    @abstractmethod
    def get_entity(self, entity_id: int) -> Optional[dict]:
        pass

    @abstractmethod
    def create_entity(self, entity: dict) -> dict:
        """Create a new entity and return it"""
        pass

    @abstractmethod
    def create_rel(self, from_id: int, to_id: int, predicate: str) -> None:
        pass

    @abstractmethod
    def delete_entity(self, entity_id: int) -> None:
        """Delete a given entity and its relations"""
        pass

    def set_property(self, entity_id: int, property_name: str, property_value: Any):
        pass
