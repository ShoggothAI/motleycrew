import functools
import inspect
from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from langchain.tools import BaseTool, StructuredTool
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

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

import asyncio

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.common import logger
from motleycrew.common.types import MotleySupportedTool
from motleycrew.common.utils import ensure_module_is_installed


class DirectOutput(BaseException):
    """Auxiliary exception to return a tool's output directly.

    When the tool returns an output, this exception is raised with the output.
    It is then handled by the agent, who should gracefully return the output to the user.
    """

    def __init__(self, output: Any):
        self.output = output


@dataclass
class RetryConfig:
    """Configuration for retry behavior of MotleyTool.

    Attributes:
        max_retries (int): Maximum number of retry attempts.
        wait_time (float): Base wait time between retries in seconds.
        backoff_factor (float): Multiplicative factor for exponential backoff.
        exceptions_to_retry (List[Type[Exception]]): Exceptions that should trigger a retry.
    """

    max_retries: int = 3
    wait_time: float = 1.0
    backoff_factor: float = 2.0
    exceptions_to_retry: Tuple[Type[Exception]] = (Exception,)


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
        exceptions_to_reflect: Optional[List[Type[Exception]]] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the MotleyTool.

        Args:
            name: Name of the tool (required if tool is None).
            description: Description of the tool (required if tool is None).
            args_schema: Schema of the tool arguments (required if tool is None).
            tool: Langchain BaseTool to wrap.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.
            retry_config: Configuration for retry behavior. If None, exceptions will not be retried.
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

        self.retry_config = retry_config or RetryConfig(max_retries=0, exceptions_to_retry=())

        self._patch_tool_run()
        if self.is_async:
            self._patch_tool_arun()

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

    @property
    def is_async(self):
        """Check if the tool is asynchronous."""
        return getattr(self.tool, "coroutine", None) is not None

    def _patch_tool_run(self):
        """Patch the tool run method to implement retry logic and reflect exceptions."""

        original_run = self.tool._run
        signature = inspect.signature(original_run)

        @functools.wraps(original_run)
        def patched_run(*args, **kwargs):
            for attempt in range(self.retry_config.max_retries + 1):
                try:
                    result = original_run(*args, **kwargs)
                    if self.return_direct:
                        raise DirectOutput(result)
                    else:
                        return result
                except Exception as e:
                    e_repr = f"{e.__class__.__name__}: {e}"

                    if attempt < self.retry_config.max_retries and isinstance(
                        e, self.retry_config.exceptions_to_retry
                    ):
                        logger.info(
                            f"Retry {attempt + 1} of {self.retry_config.max_retries} in tool {self.name}: {e_repr}"
                        )
                        sleep(
                            self.retry_config.wait_time
                            * (self.retry_config.backoff_factor**attempt)
                        )
                    else:
                        if any(isinstance(e, exc_type) for exc_type in self.exceptions_to_reflect):
                            logger.info(f"Reflecting exception in tool {self.name}: {e_repr}")
                            return e_repr
                        raise e

        patched_run.__signature__ = signature
        object.__setattr__(self.tool, "_run", patched_run)

    def _patch_tool_arun(self):
        """Patch the tool arun method to implement retry logic and reflect exceptions."""
        original_arun = self.tool._arun
        signature = inspect.signature(original_arun)

        @functools.wraps(original_arun)
        async def patched_arun(*args, **kwargs):
            for attempt in range(self.retry_config.max_retries + 1):
                try:
                    result = await original_arun(*args, **kwargs)
                    if self.return_direct:
                        raise DirectOutput(result)
                    else:
                        return result
                except Exception as e:
                    e_repr = f"{e.__class__.__name__}: {e}"

                    if attempt < self.retry_config.max_retries and isinstance(
                        e, self.retry_config.exceptions_to_retry
                    ):
                        logger.info(
                            f"Retry {attempt + 1} of {self.retry_config.max_retries} in tool {self.name}: {e_repr}"
                        )
                        await asyncio.sleep(
                            self.retry_config.wait_time
                            * (self.retry_config.backoff_factor**attempt)
                        )
                    else:
                        if any(isinstance(e, exc_type) for exc_type in self.exceptions_to_reflect):
                            logger.info(f"Reflecting exception in tool {self.name}: {e_repr}")
                            return e_repr
                        raise e

        patched_arun.__signature__ = signature
        object.__setattr__(self.tool, "_arun", patched_arun)

    async def ainvoke(
        self,
        input: Union[str, Dict],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        return await self.tool.ainvoke(input=input, config=config, **kwargs)


    def invoke(
        self,
        input: Union[str, Dict],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        return self.tool.invoke(input=input, config=config, **kwargs)

    def run(self, *args, **kwargs):
        pass

    async def arun(self, *args, **kwargs):
        pass

    def _tool_from_run_method(self, name: str, description: str, args_schema: BaseModel):
        func = None
        coroutine = None

        if self.__class__.run != MotleyTool.run:
            func = self.run
        if self.__class__.arun != MotleyTool.arun:
            coroutine = self.arun

        if func is None and coroutine is None:
            raise Exception(
                "At least one of run and arun methods must be overridden in MotleyTool if not "
                "constructing from a supported tool instance."
            )

        return StructuredTool.from_function(
            name=name,
            description=description,
            args_schema=args_schema,
            func=func,
            coroutine=coroutine,
        )

    @staticmethod
    def from_langchain_tool(
        langchain_tool: BaseTool,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> "MotleyTool":
        return MotleyTool(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
            retry_config=retry_config,
        )

    @staticmethod
    def from_llama_index_tool(
        llama_index_tool: LlamaIndex__BaseTool,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> "MotleyTool":
        """Create a MotleyTool from a LlamaIndex tool.

        Args:
            llama_index_tool: LlamaIndex tool to convert.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.
            retry_config: Configuration for retry behavior. If None, exceptions will not be retried.

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
            retry_config=retry_config,
        )

    @staticmethod
    def from_crewai_tool(
        crewai_tool: CrewAI__BaseTool,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> "MotleyTool":
        """Create a MotleyTool from a CrewAI tool.

        Args:
            crewai_tool: CrewAI tool to convert.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.
            retry_config: Configuration for retry behavior. If None, exceptions will not be retried.

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
            retry_config=retry_config,
        )

    @staticmethod
    def from_motley_agent(
        agent: MotleyAgentAbstractParent,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> "MotleyTool":
        """Convert an agent to a tool to be used by other agents via delegation.

        Args:
            agent: The MotleyAgent to convert to a tool.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.
            retry_config: Configuration for retry behavior. If None, exceptions will not be retried.

        Returns:
            The tool representation of the agent.
        """

        return agent.as_tool(
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
            retry_config=retry_config,
        )

    @staticmethod
    def from_supported_tool(
        tool: MotleySupportedTool,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> "MotleyTool":
        """Create a MotleyTool from any supported tool type.

        Args:
            tool: Tool of any supported type.
                Currently, we support tools from Langchain, LlamaIndex,
                as well as motleycrew agents.
            return_direct: If True, the tool's output will be returned directly to the user.
            exceptions_to_reflect: List of exceptions to reflect back to the agent.
            retry_config: Configuration for retry behavior. If None, exceptions will not be retried.
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
                retry_config=retry_config,
            )
        elif isinstance(tool, LlamaIndex__BaseTool):
            return MotleyTool.from_llama_index_tool(
                tool,
                return_direct=return_direct,
                exceptions_to_reflect=exceptions_to_reflect,
                retry_config=retry_config,
            )
        elif isinstance(tool, MotleyAgentAbstractParent):
            return MotleyTool.from_motley_agent(
                tool,
                return_direct=return_direct,
                exceptions_to_reflect=exceptions_to_reflect,
                retry_config=retry_config,
            )
        elif CrewAI__BaseTool is not None and isinstance(tool, CrewAI__BaseTool):
            return MotleyTool.from_crewai_tool(
                tool,
                return_direct=return_direct,
                exceptions_to_reflect=exceptions_to_reflect,
                retry_config=retry_config,
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

        llama_index_tool = LlamaIndex__FunctionTool.from_defaults(
            fn=fn,
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
        fields = list(self.tool.args_schema.model_fields.items())
        if len(fields) != 1:
            raise Exception("Multiple input fields are not supported in to_autogen_tool")

        field_name, field_info = fields[0]
        field_type = field_info.annotation

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
