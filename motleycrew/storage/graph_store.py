from abc import ABC, abstractmethod
from typing import Optional, Type
from motleycrew.storage import MotleyGraphNode, MotleyGraphNodeType


class MotleyGraphStore(ABC):
    """Abstract class for a graph database store."""

    @abstractmethod
    def check_node_exists_by_class_and_id(
        self, node_class: Type[MotleyGraphNode], node_id: int
    ) -> bool:
        """Check if a node of given class with given id is present in the database.

        Args:
            node_class: Python class of the node
            node_id: id of the node
        """
        pass

    @abstractmethod
    def check_node_exists(self, node: MotleyGraphNode) -> bool:
        """Check if the given node is present in the database.

        Args:
            node: node to check

        Returns:
            whether the node is present in the database
        """

        pass

    @abstractmethod
    def check_relation_exists(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: Optional[str]
    ) -> bool:
        """Check if a relation exists between two nodes with given label.

        Args:
            from_node: starting node
            to_node: ending node
            label: relation label. If None, check if any relation exists between the nodes.

        """
        pass

    @abstractmethod
    def get_node_by_class_and_id(
        self, node_class: Type[MotleyGraphNodeType], node_id: int
    ) -> Optional[MotleyGraphNodeType]:
        """ Retrieve the node of given class with given id if it is present in the database.
        Otherwise, return None.

        Args:
            node_class (Type[MotleyGraphNodeType]):
            node_id (int):

        Returns:
            :obj:`MotleyGraphNodeType`, None:
        """
        pass

    @abstractmethod
    def insert_node(self, node: MotleyGraphNode):
        """ Insert a new node, populate its id and freeze it.
        If node table or some columns do not exist, this method also creates them.

        Args:
            node (MotleyGraphNode):

        Returns:

        """
        pass

    @abstractmethod
    def create_relation(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str
    ) -> None:
        """ Create a relation with given label between existing nodes.
        If relation table does not exist, this method also creates them.

        Args:
            from_node (MotleyGraphNode):
            to_node (MotleyGraphNode):
            label (str):

        Returns:

        """
        pass

    @abstractmethod
    def upsert_triplet(self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str):
        """ Create a relation with given label between nodes, if such does not already exist.
        If the nodes do not already exist, create them too.
        This method also creates and/or updates all necessary tables.

        Args:
            from_node (MotleyGraphNode):
            to_node (MotleyGraphNode):
            label (str):

        Returns:

        """
        pass

    @abstractmethod
    def delete_node(self, node: MotleyGraphNode) -> None:
        """ Delete a given node and its relations.

        Args:
            node (MotleyGraphNode):

        Returns:

        """
        pass

    @abstractmethod
    def update_property(self, node: MotleyGraphNode, property_name: str):
        """ Update a graph node's property with the corresponding value from the node object.

        Args:
            node (MotleyGraphNode):
            property_name (str):

        Returns:

        """
        pass

    def ensure_node_table(self, node_class: Type[MotleyGraphNode]) -> str:  # for Kuzu
        """ Create a table for storing nodes of that class if such does not already exist.
        If it does exist, create all missing columns.

        Args:
            node_class (Type[MotleyGraphNode]):
        Returns:
            str: Table name
        """
        pass

    def ensure_relation_table(
        self, from_class: Type[MotleyGraphNode], to_class: Type[MotleyGraphNode], label: str
    ):  # for Kuzu
        """ Create a table for storing relations from from_node-like nodes to to_node-like nodes,
        if such does not already exist.

        Args:
            from_class (Type[MotleyGraphNode]):
            to_class (Type[MotleyGraphNode]):
            label (str):

        Returns:

        """
        pass

    def run_cypher_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
        container: Optional[Type[MotleyGraphNodeType]] = None,
    ) -> list[list | MotleyGraphNodeType]:
        """ Run a Cypher query and return the results.
        If container class is provided, deserialize the results into objects of that class.

        Args:
            query (:obj:`dict`, None):
            parameters (:obj:`dict`, optional):
            container (:obj:`Type[MotleyGraphNodeType]`, optional):

        Returns:

        """
        pass
