""" Module description """
from typing import Optional, Type, Callable, Any

from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, create_model

try:
    from autogen import ConversableAgent, ChatResult
except ImportError:
    ConversableAgent = None
    ChatResult = None

from motleycrew.tools import MotleyTool
from motleycrew.common.utils import ensure_module_is_installed


def get_last_message(chat_result: ChatResult) -> str:
    """ Description

    Args:
        chat_result (ChatResult):

    Returns:
        str:
    """
    for message in reversed(chat_result.chat_history):
        if message.get("content") and "TERMINATE" not in message["content"]:
            return message["content"]


class AutoGenChatTool(MotleyTool):
    def __init__(
        self,
        name: str,
        description: str,
        prompt: str | BasePromptTemplate,
        initiator: ConversableAgent,
        recipient: ConversableAgent,
        result_extractor: Callable[[ChatResult], Any] = get_last_message,
        input_schema: Optional[Type[BaseModel]] = None,
    ):
        """ Description

        Args:
            name (str):
            description (str):
            prompt (:obj:`str`, :obj:`BasePromptTemplate`):
            initiator (ConversableAgent):
            recipient (ConversableAgent):
            result_extractor (:obj:`Callable[[ChatResult]`, :obj:`Any`, optional):
            input_schema (:obj:`Type[BaseModel]`, optional):
        """
        ensure_module_is_installed("autogen")
        langchain_tool = create_autogen_chat_tool(
            name=name,
            description=description,
            prompt=prompt,
            initiator=initiator,
            recipient=recipient,
            result_extractor=result_extractor,
            input_schema=input_schema,
        )
        super().__init__(langchain_tool)


def create_autogen_chat_tool(
    name: str,
    description: str,
    prompt: str | BasePromptTemplate,
    initiator: ConversableAgent,
    recipient: ConversableAgent,
    result_extractor: Callable[[ChatResult], Any],
    input_schema: Optional[Type[BaseModel]] = None,
):
    """ Description

    Args:
        name (str):
        description (str):
        prompt (:obj:`str`, :obj:`BasePromptTemplate`):
        initiator (ConversableAgent):
        recipient (ConversableAgent):
        result_extractor (:obj:`Callable[[ChatResult]`, :obj:`Any`, optional):
        input_schema (:obj:`Type[BaseModel]`, optional):

    Returns:

    """
    if not isinstance(prompt, BasePromptTemplate):
        prompt = PromptTemplate.from_template(prompt)

    if input_schema is None:
        fields = {
            var: (str, Field(description=f"Input {var} for the tool."))
            for var in prompt.input_variables
        }

        # Create the AutoGenChatToolInput class dynamically
        input_schema = create_model("AutoGenChatToolInput", **fields)

    def run_autogen_chat(**kwargs) -> Any:
        message = prompt.format(**kwargs)
        chat_result = initiator.initiate_chat(recipient, message=message)
        return result_extractor(chat_result)

    return StructuredTool.from_function(
        func=run_autogen_chat,
        name=name,
        description=description,
        args_schema=input_schema,
    )
