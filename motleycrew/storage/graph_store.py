from abc import ABC, abstractmethod
from typing import Optional, Any, Type, TypeVar
from pydantic import BaseModel


ModelType = TypeVar("ModelType", bound=BaseModel)


class MotleyGraphStore(ABC):
    @abstractmethod
    def check_entity_exists_by_class_and_id(
        self, entity_class: Type[ModelType], entity_id: int
    ) -> bool:
        """
        Check if an entity of given class with given id is present in the database.
        """
        pass

    @abstractmethod
    def check_entity_exists(self, entity: ModelType) -> bool:
        """
        Check if the given entity is present in the database.
        """
        pass

    @abstractmethod
    def check_relation_exists(
        self, from_entity: ModelType, to_entity: ModelType, predicate: Optional[str]
    ) -> bool:
        """
        Check if a relation exists between two entities with given predicate.
        """
        pass

    @abstractmethod
    def get_entity_by_class_and_id(
        self, entity_class: Type[ModelType], entity_id: int
    ) -> Optional[ModelType]:
        """
        Retrieve the entity of given class with given id if it is present in the database.
        Otherwise, return None.
        """
        pass

    @abstractmethod
    def create_entity(self, entity: ModelType):
        """
        Create a new entity, populate its id and freeze it.
        If entity table or some columns do not exist, this method also creates them.
        """
        pass

    @abstractmethod
    def create_relation(self, from_entity: ModelType, to_entity: ModelType, predicate: str) -> None:
        """
        Create a relation between existing entities.
        If relation table does not exist, this method also creates them.
        """
        pass

    @abstractmethod
    def upsert_triplet(self, from_entity: ModelType, to_entity: ModelType, predicate: str):
        """
        Create a relation with a given predicate between entities, if such does not already exist.
        If the entities do not already exist, create them too.
        This method also creates and/or updates all necessary tables.
        """
        pass

    @abstractmethod
    def delete_entity(self, entity: ModelType) -> None:
        """
        Delete a given entity and its relations.
        """
        pass

    @abstractmethod
    def set_property(self, entity: ModelType, property_name: str, property_value: Any):
        """
        Set a property to an entity. Also sets the property in the Python object.
        """
        pass

    def run_cypher_query(self, query: str, parameters: Optional[dict] = None) -> list[list]:
        """
        Run a Cypher query and return the results.
        """
        pass
