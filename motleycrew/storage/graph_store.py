from abc import ABC, abstractmethod
from typing import Optional, Any


class MotleyGraphStore(ABC):
    node_table_name: str
    rel_table_name: str

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

    @abstractmethod
    def set_property(self, entity_id: int, property_name: str, property_value: Any):
        pass

    @abstractmethod
    def run_query(self, query: str, parameters: Optional[dict] = None) -> list[list]:
        """Run a Cypher query and return the results in standard Python containers"""
        pass
