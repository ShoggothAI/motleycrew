from typing import Optional, Any, TypeVar, TYPE_CHECKING
from abc import ABC
from pydantic import BaseModel

if TYPE_CHECKING:
    from motleycrew.storage import MotleyGraphStore


class MotleyGraphNode(BaseModel, ABC):
    """Base class for describing nodes in the graph.

    Attributes:
        __label__: Label of the node in the graph. If not set, the class name is used.
        __graph_store__: Graph store in which the node is stored.
    """

    # Q: KuzuGraphNode a better name? Because def id is specific?
    # A: No, I think _id attribute is pretty universal
    __label__: Optional[str] = None
    __graph_store__: Optional["MotleyGraphStore"] = None

    @property
    def id(self) -> Optional[Any]:
        """Identifier of the node in the graph.

        The identifier is unique **among nodes of the same label**.
        If the node is not inserted in the graph, the identifier is None.
        """
        return getattr(self, "_id", None)

    @property
    def is_inserted(self) -> bool:
        """Whether the node is inserted in the graph."""
        return self.id is not None

    @classmethod
    def get_label(cls) -> str:
        """Get the label of the node.

        Labels can be viewed as node types in the graph.
        Generally, the label is the class name,
        but it can be overridden by setting the __label__ attribute.

        Returns:
            Label of the node.
        """

        # Q: why not @property def label(cls) -> str: return cls.__label__ or cls.__name__ ?
        # A: Because we want to be able to call this method without an instance
        #    and properties can't be class methods since Python 3.12
        if cls.__label__:
            return cls.__label__
        return cls.__name__

    def __setattr__(self, name, value):
        """Set the attribute value
        and update the property in the graph store if the node is inserted.

        Args:
            name: Name of the attribute.
            value: Value of the attribute.
        """
        super().__setattr__(name, value)

        if name not in self.model_fields:
            # Q: Should we not raise an error here instead?
            # A: No, there are technical attributes like __graph_store__ that are not in the model
            return  # Non-pydantic field => not in the DB

        if self.__graph_store__ and self.is_inserted:
            self.__graph_store__.update_property(self, name)

    def __eq__(self, other):
        """Comparison operator for nodes.

        Two nodes are considered equal if they have the same label and identifier.

        Args:
            other: Node to compare with.

        Returns:
            Whether the nodes are equal.
        """
        return self.is_inserted and self.get_label() == other.get_label() and self.id == other.id


MotleyGraphNodeType = TypeVar("MotleyGraphNodeType", bound=MotleyGraphNode)
