""" Module description """
from motleycrew.common import LLMFamily
from motleycrew.common import GraphStoreType


class Defaults:
    """ Description

    Attributes:
        DEFAULT_LLM_FAMILY (str):
        DEFAULT_LLM_NAME (str):
        DEFAULT_LLM_TEMPERATURE (float):
        LLM_MAP (dict):
        DEFAULT_GRAPH_STORE_TYPE (str):
        MODULE_INSTALL_COMMANDS (dict):
        DEFAULT_NUM_THREADS (int):
        DEFAULT_EVENT_LOOP_SLEEP (int):

    """
    DEFAULT_LLM_FAMILY = LLMFamily.OPENAI
    DEFAULT_LLM_NAME = "gpt-4-turbo"
    DEFAULT_LLM_TEMPERATURE = 0.0
    LLM_MAP = {}

    DEFAULT_GRAPH_STORE_TYPE = GraphStoreType.KUZU

    MODULE_INSTALL_COMMANDS = {
        "crewai": "pip install crewai",
        "llama_index": "pip install llama-index",
        "autogen": "pip install autogen",
    }
    DEFAULT_NUM_THREADS = 4
    DEFAULT_EVENT_LOOP_SLEEP = 1
