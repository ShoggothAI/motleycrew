""" Module description """

from typing import Union, Optional, Dict, Any
from typing import Callable

from langchain.tools import BaseTool
from langchain_core.runnables import Runnable, RunnableConfig

try:
    from llama_index.core.tools import BaseTool as LlamaIndex__BaseTool
    from llama_index.core.tools import FunctionTool as LlamaIndex__FunctionTool
except ImportError:
    LlamaIndex__BaseTool = None
    LlamaIndex__FunctionTool = None

from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.common.types import MotleySupportedTool
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent


class MotleyTool(Runnable):

    def __init__(self, tool: BaseTool):
        """Base tool class compatible with MotleyAgents.
        It is a wrapper for LangChain's BaseTool, containing all necessary adapters and converters.

        Args:
            tool (BaseTool):
        """
        self.tool = tool

    def __repr__(self):
        return f"MotleyTool(name={self.name})"

    def __str__(self):
        return self.__repr__()

    @property
    def name(self):
        # TODO: do we really want to make a thin wrapper in this fashion?
        return self.tool.name

    @property
    def description(self):
        return self.tool.description

    @property
    def args_schema(self):
        return self.tool.args_schema

    def invoke(
        self,
        input: Union[str, Dict],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        return self.tool.invoke(input=input, config=config, **kwargs)

    def _run(self, *args: tuple, **kwargs: Dict[str, Any]) -> Any:
        return self.tool._run(*args, **kwargs)

    @staticmethod
    def from_langchain_tool(langchain_tool: BaseTool) -> "MotleyTool":
        """Description

        Args:
            langchain_tool (BaseTool):

        Returns:
            MotleyTool:
        """
        return MotleyTool(tool=langchain_tool)

    @staticmethod
    def from_llama_index_tool(llama_index_tool: LlamaIndex__BaseTool) -> "MotleyTool":
        """Description

        Args:
            llama_index_tool (LlamaIndex__BaseTool):

        Returns:
            MotleyTool:
        """
        ensure_module_is_installed("llama_index")
        langchain_tool = llama_index_tool.to_langchain_tool()
        return MotleyTool.from_langchain_tool(langchain_tool=langchain_tool)

    @staticmethod
    def from_supported_tool(tool: MotleySupportedTool) -> "MotleyTool":
        """Description

        Args:
            tool (:obj:`MotleyTool`, :obj:`BaseTool`, :obj:`LlamaIndex__BaseTool`, :obj:`MotleyAgentAbstractParent`):

        Returns:

        """
        if isinstance(tool, MotleyTool):
            return tool
        elif isinstance(tool, BaseTool):
            return MotleyTool.from_langchain_tool(tool)
        elif isinstance(tool, LlamaIndex__BaseTool):
            return MotleyTool.from_llama_index_tool(tool)
        elif isinstance(tool, MotleyAgentAbstractParent):
            return tool.as_tool()
        else:
            raise Exception(
                f"Tool type `{type(tool)}` is not supported, please convert to MotleyTool first"
            )

    def to_langchain_tool(self) -> BaseTool:
        """Description

        Returns:
            BaseTool:
        """
        return self.tool

    def to_llama_index_tool(self) -> LlamaIndex__BaseTool:
        """Description

        Returns:
            LlamaIndex__BaseTool:
        """
        ensure_module_is_installed("llama_index")
        llama_index_tool = LlamaIndex__FunctionTool.from_defaults(
            fn=self.tool._run,
            name=self.tool.name,
            description=self.tool.description,
            fn_schema=self.tool.args_schema,
        )
        return llama_index_tool

    def to_autogen_tool(self) -> Callable:
        """Description

        Returns:
            Callable:
        """
        fields = list(self.tool.args_schema.__fields__.values())
        if len(fields) != 1:
            raise Exception("Multiple input fields are not supported in to_autogen_tool")

        field_name = fields[0].name
        field_type = fields[0].annotation

        def autogen_tool_fn(input: field_type) -> str:
            return self.invoke({field_name: input})

        return autogen_tool_fn
