""" Module description """
from typing import Sequence

try:
    from llama_index.core.agent import ReActAgent
    from llama_index.core.llms import LLM
    from llama_index.core.callbacks import CallbackManager
except ImportError:
    LLM = object

from motleycrew.agents.llama_index import LlamaIndexMotleyAgent
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.tools import MotleyTool
from motleycrew.common import MotleySupportedTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.tracking import get_default_callbacks_list
from motleycrew.common.utils import ensure_module_is_installed


class ReActLlamaIndexMotleyAgent(LlamaIndexMotleyAgent):
    def __init__(
        self,
        description: str,
        name: str | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        llm: LLM | None = None,
        verbose: bool = False,
    ):
        """ Description

        Args:
            description (str):
            name (:obj:`str`, optional):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            llm (:obj:`LLM`, optional):
            verbose (:obj:`bool`, optional):
        """
        ensure_module_is_installed("llama_index")
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LLAMA_INDEX)

        def agent_factory(tools: dict[str, MotleyTool]):
            llama_index_tools = [t.to_llama_index_tool() for t in tools.values()]
            # TODO: feed description into the agent's prompt
            callbacks = get_default_callbacks_list(LLMFramework.LLAMA_INDEX)
            agent = ReActAgent.from_tools(
                tools=llama_index_tools,
                llm=llm,
                verbose=verbose,
                callback_manager=CallbackManager(callbacks),
            )
            return agent

        super().__init__(
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            verbose=verbose,
        )
