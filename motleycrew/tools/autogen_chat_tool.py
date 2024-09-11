from typing import Optional, Type, Callable, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

try:
    from autogen import ConversableAgent, ChatResult
except ImportError:
    ConversableAgent = None
    ChatResult = None

from motleycrew.tools import MotleyTool
from motleycrew.common.utils import ensure_module_is_installed


def get_last_message(chat_result: ChatResult) -> str:
    for message in reversed(chat_result.chat_history):
        if message.get("content") and "TERMINATE" not in message["content"]:
            return message["content"]


class AutoGenChatTool(MotleyTool):
    """A tool for incorporating AutoGen chats into MotleyCrew."""

    def __init__(
        self,
        name: str,
        description: str,
        prompt: str | BasePromptTemplate,
        initiator: ConversableAgent,
        recipient: ConversableAgent,
        result_extractor: Callable[[ChatResult], Any] = get_last_message,
        input_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        """
        Args:
            name: Name of the tool.
            description: Description of the tool.
            prompt: Prompt to use for the tool. Can be a string or a PromptTemplate object.
            initiator: The agent initiating the chat.
            recipient: The first recipient agent.
                This is the agent that you would specify in ``initiate_chat`` arguments.
            result_extractor: Function to extract the result from the chat result.
            input_schema: Input schema for the tool.
                The input variables should match the variables in the prompt.
                If not provided, a schema will be generated based on the input variables
                in the prompt, if any, with string fields.
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
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


def create_autogen_chat_tool(
    name: str,
    description: str,
    prompt: str | BasePromptTemplate,
    initiator: ConversableAgent,
    recipient: ConversableAgent,
    result_extractor: Callable[[ChatResult], Any],
    input_schema: Optional[Type[BaseModel]] = None,
):
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
