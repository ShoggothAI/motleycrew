from motleycrew.common.enums import GraphStoreType
from motleycrew.common.enums import LLMProvider


class Defaults:
    """Default values for various settings."""

    DEFAULT_REACT_AGENT_MAX_ITERATIONS = 15
    DEFAULT_LLM_PROVIDER = LLMProvider.OPENAI
    DEFAULT_LLM_NAME = "gpt-4o"
    DEFAULT_LLM_TEMPERATURE = 0.0

    DEFAULT_GRAPH_STORE_TYPE = GraphStoreType.KUZU

    MODULE_INSTALL_COMMANDS = {
        "crewai": "pip install crewai",
        "llama_index": "pip install llama-index",
        "autogen": "pip install pyautogen",
        "lunary": "pip install lunary",
        "aider": "pip install aider-chat",
        "pglast": "pip install pglast",
        "crewai_tools": "pip install 'crewai[tools]'",
    }

    DEFAULT_NUM_THREADS = 4
    DEFAULT_EVENT_LOOP_SLEEP = 1
    DEFAULT_OUTPUT_HANDLER_MAX_ITERATIONS = 5
