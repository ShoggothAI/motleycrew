import functools
import inspect
from typing import Callable, Union, Optional, Dict, Any, List

from langchain.tools import BaseTool, Tool, StructuredTool
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.pydantic_v1 import BaseModel

from motleycrew.common.exceptions import InvalidOutput

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

from motleycrew.common import logger
from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.common.types import MotleySupportedTool
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent


class DirectOutput(BaseException):
    """Auxiliary exception to return a tool's output directly.

    When the tool returns an output, this exception is raised with the output.
    It is then handled by the agent, who should gracefully return the output to the user.
    """

    def __init__(self, output: Any):
        self.output = output


class MotleyTool(Runnable):
    """Base tool class compatible with MotleyAgents.

    It is a wrapper for Langchain BaseTool, containing all necessary adapters and converters.
    """

    def __init__(
        self,
        tool: Optional[BaseTool] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[BaseModel] = None,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        """Initialize the MotleyTool.

        Args:
            name: Name of the tool (required if tool is None).
            description: Description of the tool (required if tool is None).
            args_schema: Schema of the tool arguments (required if tool is None).
            tool: Langchain BaseTool to wrap.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.
        """
        if tool is None:
            assert name is not None
            assert description is not None
            self.tool = self._tool_from_run_method(
                name=name, description=description, args_schema=args_schema
            )
        else:
            self.tool = tool

        self.return_direct = return_direct
        self.exceptions_to_reflect = exceptions_to_reflect or []
        if InvalidOutput not in self.exceptions_to_reflect:
            self.exceptions_to_reflect = [InvalidOutput, *self.exceptions_to_reflect]

        self._patch_tool_run()

        self.agent: Optional[MotleyAgentAbstractParent] = None
        self.agent_input: Optional[dict] = None

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

    def _patch_tool_run(self):
        """Patch the tool run method to reflect exceptions."""

        original_run = self.tool._run
        signature = inspect.signature(original_run)

        @functools.wraps(original_run)
        def patched_run(*args, **kwargs):
            try:
                result = original_run(*args, **kwargs)
                if self.return_direct:
                    raise DirectOutput(result)
                else:
                    return result
            except tuple(self.exceptions_to_reflect or []) as e:
                # we need to return the exception to the agent
                return f"{e.__class__.__name__}: {e}"

        patched_run.__signature__ = signature
        object.__setattr__(self.tool, "_run", patched_run)

    def invoke(
        self,
        input: Union[str, Dict],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        return self.tool.invoke(input=input, config=config, **kwargs)

    def run(self, *args, **kwargs):
        pass

    def _tool_from_run_method(self, name: str, description: str, args_schema: BaseModel):
        return StructuredTool.from_function(
            name=name,
            description=description,
            args_schema=args_schema,
            func=self.run,
        )

    @staticmethod
    def from_langchain_tool(
        langchain_tool: BaseTool,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ) -> "MotleyTool":
        """Create a MotleyTool from a Langchain tool.

        Args:
            langchain_tool: Langchain tool to convert.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.

        Returns:
            MotleyTool instance.
        """
        if langchain_tool.return_direct:
            logger.warning(
                "Please set `return_direct` in MotleyTool instead of the tool you're converting. "
                "Automatic conversion will be removed in motleycrew v1."
            )
            return_direct = True
            langchain_tool.return_direct = False

        return MotleyTool(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )

    @staticmethod
    def from_llama_index_tool(
        llama_index_tool: LlamaIndex__BaseTool,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ) -> "MotleyTool":
        """Create a MotleyTool from a LlamaIndex tool.

        Args:
            llama_index_tool: LlamaIndex tool to convert.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.

        Returns:
            MotleyTool instance.
        """
        ensure_module_is_installed("llama_index")
        if llama_index_tool.metadata and llama_index_tool.metadata.return_direct:
            logger.warning(
                "Please set `return_direct` in MotleyTool instead of the tool you're converting. "
                "Automatic conversion will be removed in motleycrew v1."
            )
            return_direct = True
            llama_index_tool.metadata.return_direct = False

        langchain_tool = llama_index_tool.to_langchain_tool()
        return MotleyTool.from_langchain_tool(
            langchain_tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )

    @staticmethod
    def from_crewai_tool(
        crewai_tool: CrewAI__BaseTool,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ) -> "MotleyTool":
        """Create a MotleyTool from a CrewAI tool.

        Args:
            crewai_tool: CrewAI tool to convert.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.

        Returns:
            MotleyTool instance.
        """
        ensure_module_is_installed("crewai_tools")
        langchain_tool = crewai_tool.to_langchain()

        # change tool name punctuation
        for old_symbol, new_symbol in [(" ", "_"), ("'", "")]:
            langchain_tool.name = langchain_tool.name.replace(old_symbol, new_symbol)

        return MotleyTool.from_langchain_tool(
            langchain_tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )

    @staticmethod
    def from_motley_agent(
        agent: MotleyAgentAbstractParent,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ) -> "MotleyTool":
        """Convert an agent to a tool to be used by other agents via delegation.

        Returns:
            The tool representation of the agent.
        """

        return agent.as_tool(
            return_direct=return_direct, exceptions_to_reflect=exceptions_to_reflect
        )

    @staticmethod
    def from_supported_tool(
        tool: MotleySupportedTool,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ) -> "MotleyTool":
        """Create a MotleyTool from any supported tool type.

        Args:
            tool: Tool of any supported type.
                Currently, we support tools from Langchain, LlamaIndex,
                as well as motleycrew agents.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.
        Returns:
            MotleyTool instance.
        """
        if isinstance(tool, MotleyTool):
            return tool
        elif isinstance(tool, BaseTool):
            return MotleyTool.from_langchain_tool(
                tool,
                return_direct=return_direct,
                exceptions_to_reflect=exceptions_to_reflect,
            )
        elif isinstance(tool, LlamaIndex__BaseTool):
            return MotleyTool.from_llama_index_tool(
                tool,
                return_direct=return_direct,
                exceptions_to_reflect=exceptions_to_reflect,
            )
        elif isinstance(tool, MotleyAgentAbstractParent):
            return MotleyTool.from_motley_agent(
                tool,
                return_direct=return_direct,
                exceptions_to_reflect=exceptions_to_reflect,
            )
        elif CrewAI__BaseTool is not None and isinstance(tool, CrewAI__BaseTool):
            return MotleyTool.from_crewai_tool(
                tool,
                return_direct=return_direct,
                exceptions_to_reflect=exceptions_to_reflect,
            )
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

        if inspect.signature(self.tool._run).parameters.get("config", None) is not None:
            fn = functools.partial(self.tool._run, config=RunnableConfig())
        else:
            fn = self.tool._run

        fn_schema = self.tool.args_schema
        fn_schema.model_json_schema = (
            fn_schema.schema
        )  # attempt to make it compatible with Langchain's old Pydantic v1

        llama_index_tool = LlamaIndex__FunctionTool.from_defaults(
            fn=fn,
            name=self.tool.name,
            description=self.tool.description,
            fn_schema=fn_schema,
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
