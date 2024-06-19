""" Module description"""
from typing import Any, Optional

from motleycrew.common import Defaults


class LLMFamilyNotSupported(Exception):
    """ Description

    Args:
        llm_framework (str):
        llm_family (str):
    """
    def __init__(self, llm_framework: str, llm_family: str):
        self.llm_framework = llm_framework
        self.llm_family = llm_family

    def __str__(self) -> str:
        return f"LLM family `{self.llm_family}` is not supported via the framework `{self.llm_framework}`"


class LLMFrameworkNotSupported(Exception):
    def __init__(self, llm_framework: str):
        """ Description

        Args:
            llm_framework (str):
        """
        self.llm_framework = llm_framework

    def __str__(self) -> str:
        return f"LLM framework `{self.llm_framework}` is not supported"


class AgentNotMaterialized(Exception):
    """ Description

   Args:
       agent_name (str):
   """
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def __str__(self) -> str:
        return f"Agent `{self.agent_name}` is not yet materialized"


class CannotModifyMaterializedAgent(Exception):
    """ Description

    Args:
        agent_name (str):
    """
    def __init__(self, agent_name: str | None):
        self.agent_name = agent_name

    def __str__(self) -> str:
        return "Cannot modify agent{} as it is already materialized".format(
            f" `{self.agent_name}`" if self.agent_name is not None else ""
        )


class TaskDependencyCycleError(Exception):
    """Raised when a task is set to depend on itself"""


class IntegrationTestException(Exception):
    """ Integration tests exception

    Args:
        test_names (list[str]): list of names of failed integration tests
    """

    def __init__(self, test_names: list[str]):
        self.test_names = test_names

    def __str__(self):
        return "Some integration tests failed: {}".format(self.test_names)


class IpynbIntegrationTestResultNotFound(Exception):
    """ Ipynb integration test not found result file exception

    Args:
        ipynb_path (str): path to running ipynb
        result_path (str): path to execution result file
    """
    def __init__(self, ipynb_path: str, result_path: str):
        self.ipynb_path = ipynb_path
        self.result_path = result_path

    def __str__(self):
        return "File result {} of the ipynb {} execution, not found.".format(self.result_path, self.ipynb_path)


class ModuleNotInstalled(Exception):
    """Module not installed

    Args:
        module_name (str): the name of the non-installed module
        install_command (:obj:`str`, optional): the command to install
    """

    def __init__(self, module_name: str, install_command: str = None):
        self.module_name = module_name
        self.install_command = install_command or Defaults.MODULE_INSTALL_COMMANDS.get(
            module_name, None
        )

    def __str__(self):
        msg = "{} is not installed".format(self.module_name)

        if self.install_command is not None:
            msg = "{}, {}".format(msg, self.install_command)

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
    """Raised in output handlers when an agent's output is not accepted"""
    pass
