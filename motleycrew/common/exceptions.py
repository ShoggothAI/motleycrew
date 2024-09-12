"""Exceptions for motleycrew"""

from typing import Any, Dict, Optional

from motleycrew.common import Defaults


class LLMProviderNotSupported(Exception):
    """Raised when an LLM provider is not supported in motleycrew via a framework."""

    def __init__(self, llm_framework: str, llm_provider: str):
        self.llm_framework = llm_framework
        self.llm_provider = llm_provider

    def __str__(self) -> str:
        return f"LLM provider `{self.llm_provider}` is not supported via the framework `{self.llm_framework}`"


class LLMFrameworkNotSupported(Exception):
    """Raised when an LLM framework is not supported in motleycrew."""

    def __init__(self, llm_framework: str):
        self.llm_framework = llm_framework

    def __str__(self) -> str:
        return f"LLM framework `{self.llm_framework}` is not supported"


class AgentNotMaterialized(Exception):
    """Raised when an attempt is made to use an agent that is not yet materialized."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def __str__(self) -> str:
        return f"Agent `{self.agent_name}` is not yet materialized"


class CannotModifyMaterializedAgent(Exception):
    """Raised when an attempt is made to modify a materialized agent, e.g. to add tools."""

    def __init__(self, agent_name: str | None):
        self.agent_name = agent_name

    def __str__(self) -> str:
        return "Cannot modify agent{} as it is already materialized".format(
            f" '{self.agent_name}'" if self.agent_name is not None else ""
        )


class TaskDependencyCycleError(Exception):
    """Raised when a task is set to depend on itself"""


class IntegrationTestException(Exception):
    """One or more integration tests failed."""

    def __init__(self, test_names: list[str]):
        """
        Args:
            test_names: List of names of failed integration tests.
        """
        self.test_names = test_names

    def __str__(self):
        return "Some integration tests failed: {}".format(self.test_names)


class IpynbIntegrationTestResultNotFound(Exception):
    """Raised when the result file of an ipynb integration test run is not found."""

    def __init__(self, ipynb_path: str, result_path: str):
        self.ipynb_path = ipynb_path
        self.result_path = result_path

    def __str__(self):
        return "File {} with result of the ipynb {} execution not found.".format(
            self.result_path, self.ipynb_path
        )


class ModuleNotInstalled(Exception):
    """Raised when trying to use some functionality that requires a module that is not installed.
    """

    def __init__(self, module_name: str, install_command: str = None):
        """
        Args:
            module_name: Name of the module.
            install_command: Command to install the module.
        """
        self.module_name = module_name
        self.install_command = install_command or Defaults.MODULE_INSTALL_COMMANDS.get(
            module_name, None
        )

    def __str__(self):
        msg = "{} is not installed".format(self.module_name)

        if self.install_command is not None:
            msg = "{}, please install ({})".format(msg, self.install_command)

        return "{}.".format(msg)


class InvalidToolInput(Exception):
    """Raised when the tool input is invalid"""

    def __init__(self, tool: Any, input: Any, message: Optional[str] = None):
        self.tool = tool
        self.input = input
        self.message = message

    def __str__(self):
        msg = "Invalid input `{}` for tool `{}`".format(self.input, self.tool_name)
        if self.message:
            msg = "{}: {}".format(msg, self.message)
        return msg


class InvalidOutput(Exception):
    """Raised in output handlers when an agent's output is not accepted."""

    pass
