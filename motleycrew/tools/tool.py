import functools
from typing import Callable, Union, Optional, Dict, Any

from langchain.tools import BaseTool
from langchain_core.runnables import Runnable, RunnableConfig

try:
    from llama_index.core.tools import BaseTool as LlamaIndex__BaseTool
    from llama_index.core.tools import FunctionTool as LlamaIndex__FunctionTool
except ImportError:
    LlamaIndex__BaseTool = None
    LlamaIndex__FunctionTool = None

try:
    from crewai_tools import BaseTool as CrewAI__BaseTool
    from crewai_tools import Tool as Crewai__Tool
except ImportError:
    CrewAI__BaseTool = None
    Crewai__Tool = None

from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.common.types import MotleySupportedTool
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent


class MotleyTool(Runnable):
    """Base tool class compatible with MotleyAgents.

    It is a wrapper for Langchain BaseTool, containing all necessary adapters and converters.
    """

    def __init__(self, tool: BaseTool):
        """Initialize the MotleyTool.

        Args:
            tool: Langchain BaseTool to wrap.
        """
        self.tool = tool

    def __repr__(self):
        return f"MotleyTool(name={self.name})"

    def __str__(self):
        return self.__repr__()

    @property
    def name(self):
        """Name of the tool."""
        return self.tool.name

    @property
    def description(self):
        """Description of the tool."""
        return self.tool.description

    @property
    def args_schema(self):
        """Schema of the tool arguments."""
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
        """Create a MotleyTool from a Langchain tool.

        Args:
            langchain_tool: Langchain tool to convert.

        Returns:
            MotleyTool instance.
        """

        return MotleyTool(tool=langchain_tool)

    @staticmethod
    def from_llama_index_tool(llama_index_tool: LlamaIndex__BaseTool) -> "MotleyTool":
        """Create a MotleyTool from a LlamaIndex tool.

        Args:
            llama_index_tool: LlamaIndex tool to convert.

        Returns:
            MotleyTool instance.
        """

        ensure_module_is_installed("llama_index")
        langchain_tool = llama_index_tool.to_langchain_tool()
        return MotleyTool.from_langchain_tool(langchain_tool=langchain_tool)

    @staticmethod
    def from_crewai_tool(crewai_tool: CrewAI__BaseTool) -> "MotleyTool":
        """Create a MotleyTool from a CrewAI tool.

        Args:
            crewai_tool: CrewAI tool to convert.

        Returns:
            MotleyTool instance.
        """
        ensure_module_is_installed("crewai_tools")
        langchain_tool = crewai_tool.to_langchain()

        # change tool name punctuation
        for old_symbol, new_symbol in [(" ", "_"), ("'", "")]:
            langchain_tool.name = langchain_tool.name.replace(old_symbol, new_symbol)

        return MotleyTool.from_langchain_tool(langchain_tool=langchain_tool)

    @staticmethod
    def from_supported_tool(tool: MotleySupportedTool) -> "MotleyTool":
        """Create a MotleyTool from any supported tool type.

        Args:
            tool: Tool of any supported type.
                Currently, we support tools from Langchain, LlamaIndex,
                as well as motleycrew agents.
        Returns:
            MotleyTool instance.
        """
        if isinstance(tool, MotleyTool):
            return tool
        elif isinstance(tool, BaseTool):
            return MotleyTool.from_langchain_tool(tool)
        elif isinstance(tool, LlamaIndex__BaseTool):
            return MotleyTool.from_llama_index_tool(tool)
        elif isinstance(tool, MotleyAgentAbstractParent):
            return tool.as_tool()
        elif CrewAI__BaseTool is not None and isinstance(tool, CrewAI__BaseTool):
            return MotleyTool.from_crewai_tool(tool)
        else:
            raise Exception(
                f"Tool type `{type(tool)}` is not supported, please convert to MotleyTool first"
            )

    def to_langchain_tool(self) -> BaseTool:
        """Convert the MotleyTool to a Langchain tool.

        Returns:
            Langchain tool.
        """
        return self.tool

    def to_llama_index_tool(self) -> LlamaIndex__BaseTool:
        """Convert the MotleyTool to a LlamaIndex tool.

        Returns:
            LlamaIndex tool.
        """
        ensure_module_is_installed("llama_index")
        llama_index_tool = LlamaIndex__FunctionTool.from_defaults(
            fn=functools.partial(self.tool._run, config=RunnableConfig()),
            name=self.tool.name,
            description=self.tool.description,
            fn_schema=self.tool.args_schema,
        )
        return llama_index_tool

    def to_autogen_tool(self) -> Callable:
        """Convert the MotleyTool to an AutoGen tool.

        An AutoGen tool is basically a function. AutoGen infers the tool input schema
        from the function signature. For this reason, because we can't generate the signature
        dynamically, we can only convert tools with a single input field.

        Returns:
            AutoGen tool function.
        """
        fields = list(self.tool.args_schema.__fields__.values())
        if len(fields) != 1:
            raise Exception("Multiple input fields are not supported in to_autogen_tool")

        field_name = fields[0].name
        field_type = fields[0].annotation

        def autogen_tool_fn(input: field_type) -> str:
            return self.invoke({field_name: input})

        return autogen_tool_fn

    def to_crewai_tool(self) -> CrewAI__BaseTool:
        """Description

        Returns:
            Crewai__BaseTool:
        """
        ensure_module_is_installed("crewai_tools")
        crewai_tool = Crewai__Tool(
            name=self.tool.name,
            description=self.tool.description,
            func=self.tool._run,
            args_schema=self.tool.args_schema,
        )
        return crewai_tool
