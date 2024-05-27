from abc import ABC, abstractmethod
from typing import Optional, Type
from motleycrew.storage import MotleyGraphNode, MotleyGraphNodeType


class MotleyGraphStore(ABC):
    @abstractmethod
    def check_node_exists_by_class_and_id(
        self, node_class: Type[MotleyGraphNode], node_id: int
    ) -> bool:
        """
        Check if a node of given class with given id is present in the database.
        """
        pass

    @abstractmethod
    def check_node_exists(self, node: MotleyGraphNode) -> bool:
        """
        Check if the given node is present in the database.
        """
        pass

    @abstractmethod
    def check_relation_exists(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: Optional[str]
    ) -> bool:
        """
        Check if a relation exists between two nodes with given label.
        """
        pass

    @abstractmethod
    def get_node_by_class_and_id(
        self, node_class: Type[MotleyGraphNodeType], node_id: int
    ) -> Optional[MotleyGraphNodeType]:
        """
        Retrieve the node of given class with given id if it is present in the database.
        Otherwise, return None.
        """
        pass

    @abstractmethod
    def insert_node(self, node: MotleyGraphNode):
        """
        Insert a new node, populate its id and freeze it.
        If node table or some columns do not exist, this method also creates them.
        """
        pass

    @abstractmethod
    def create_relation(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str
    ) -> None:
        """
        Create a relation with given label between existing nodes.
        If relation table does not exist, this method also creates them.
        """
        pass

    @abstractmethod
    def upsert_triplet(self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str):
        """
        Create a relation with given label between nodes, if such does not already exist.
        If the nodes do not already exist, create them too.
        This method also creates and/or updates all necessary tables.
        """
        pass

    @abstractmethod
    def delete_node(self, node: MotleyGraphNode) -> None:
        """
        Delete a given node and its relations.
        """
        pass

    @abstractmethod
    def update_property(self, node: MotleyGraphNode, property_name: str):
        """
        Update a graph node's property with the corresponding value from the node object.
        """
        pass

    def ensure_node_table(self, node_class: Type[MotleyGraphNode]) -> str:  # for Kuzu
        """
        Create a table for storing nodes of that class if such does not already exist.
        If it does exist, create all missing columns.
        Return the table name.
        """
        pass

    def ensure_relation_table(
        self, from_class: Type[MotleyGraphNode], to_class: Type[MotleyGraphNode], label: str
    ):  # for Kuzu
        """
        Create a table for storing relations from from_node-like nodes to to_node-like nodes,
        if such does not already exist.
        """
        pass

    def run_cypher_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
        container: Optional[Type[MotleyGraphNodeType]] = None,
    ) -> list[list | MotleyGraphNodeType]:
        """
        Run a Cypher query and return the results.
        If container class is provided, deserialize the results into objects of that class.
        """
        pass
