""" Module description

Attributes:
    MotleyGraphNodeType (TypeVar):

"""

from typing import Optional, Any, TypeVar, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from motleycrew.storage import MotleyGraphStore


class MotleyGraphNode(BaseModel):
    """Description"""

    # Q: KuzuGraphNode a better name? Because def id is specific?
    # A: No, I think _id attribute is pretty universal
    __label__: Optional[str] = None
    __graph_store__: Optional["MotleyGraphStore"] = None

    @property
    def id(self) -> Optional[Any]:
        return getattr(self, "_id", None)

    @property
    def is_inserted(self) -> bool:
        return self.id is not None

    @classmethod
    def get_label(cls) -> str:
        """Description

        Returns:
            str:
        """
        # Q: why not @property def label(cls) -> str: return cls.__label__ or cls.__name__ ?
        # A: Because we want to be able to call this method without an instance
        #    and properties can't be class methods since Python 3.12
        if cls.__label__:
            return cls.__label__
        return cls.__name__

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if name not in self.model_fields:
            # Q: Should we not raise an error here instead?
            # A: No, there are technical attributes like __graph_store__ that are not in the model
            return  # Non-pydantic field => not in the DB

        if self.__graph_store__ and self.is_inserted:
            self.__graph_store__.update_property(self, name)

    def __eq__(self, other):
        return self.is_inserted and self.get_label() == other.get_label() and self.id == other.id


MotleyGraphNodeType = TypeVar("MotleyGraphNodeType", bound=MotleyGraphNode)
