from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import StructuredTool

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.common import Defaults
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.tools import MotleyTool


class MotleyOutputHandler(MotleyTool, ABC):
    """Base class for output handler tools.

    Output handler tools are used to process the final output of an agent.

    For creating an output handler tool, inherit from this class and implement
    the `handle_output` method.

    Attributes:
        _name: Name of the output handler tool.
        _description: Description of the output handler tool.
        _args_schema: Pydantic schema for the arguments of the output handler tool.
            Inferred from the ``handle_output`` method signature if not provided.
        _exceptions_to_handle: Exceptions that should be returned to the agent when raised.
    """

    _name: str = "output_handler"
    _description: str = "Output handler. ONLY RETURN THE FINAL RESULT USING THIS TOOL!"
    _args_schema: Optional[BaseModel] = None
    _exceptions_to_handle: tuple[Exception] = (InvalidOutput,)

    def __init__(self, max_iterations: int = Defaults.DEFAULT_OUTPUT_HANDLER_MAX_ITERATIONS):
        """
        Args:
            max_iterations: Maximum number of iterations to run the output handler.
                If an exception is raised in the ``handle_output`` method, the output handler
                will return the exception to the agent unless the number of iterations exceeds
                ``max_iterations``, in which case the output handler will raise
                :class:`motleycrew.common.exceptions.OutputHandlerMaxIterationsExceeded`.
        """
        self.max_iterations = max_iterations
        langchain_tool = self._create_langchain_tool()
        super().__init__(langchain_tool)

        self.agent: Optional[MotleyAgentAbstractParent] = None
        self.agent_input: Optional[dict] = None

    @property
    def exceptions_to_handle(self):
        return self._exceptions_to_handle

    def _create_langchain_tool(self):
        return StructuredTool.from_function(
            name=self._name,
            description=self._description,
            args_schema=self._args_schema,
            func=self.handle_output,
        )

    @abstractmethod
    def handle_output(self, *args, **kwargs):
        """Method for processing the final output of an agent.

        Implement this method in your output handler tool.
        """
        pass
