"""Helper functions to initialize Language Models (LLMs) from different frameworks."""

from motleycrew.common import Defaults
from motleycrew.common import LLMFamily, LLMFramework
from motleycrew.common.exceptions import LLMFamilyNotSupported, LLMFrameworkNotSupported
from motleycrew.common.utils import ensure_module_is_installed


def langchain_openai_llm(
    llm_name: str = Defaults.DEFAULT_LLM_NAME,
    llm_temperature: float = Defaults.DEFAULT_LLM_TEMPERATURE,
    **kwargs,
):
    """Initialize an OpenAI LLM client for use with Langchain.

    Args:
        llm_name: Name of the LLM in OpenAI API.
        llm_temperature: Temperature for the LLM.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=llm_name, temperature=llm_temperature, **kwargs)


def llama_index_openai_llm(
    llm_name: str = Defaults.DEFAULT_LLM_NAME,
    llm_temperature: float = Defaults.DEFAULT_LLM_TEMPERATURE,
    **kwargs,
):
    """Initialize an OpenAI LLM client for use with LlamaIndex.

    Args:
        llm_name: Name of the LLM in OpenAI API.
        llm_temperature: Temperature for the LLM.
    """

    ensure_module_is_installed("llama_index")
    from llama_index.llms.openai import OpenAI

    return OpenAI(model=llm_name, temperature=llm_temperature, **kwargs)


def langchain_anthropic_llm(
    llm_name: str = Defaults.DEFAULT_LLM_NAME,
    llm_temperature: float = Defaults.DEFAULT_LLM_TEMPERATURE,
    **kwargs,
):
    """Initialize an Anthropic LLM client for use with Langchain.

    Args:
        llm_name: Name of the LLM in Anthropic API.
        llm_temperature: Temperature for the LLM.
    """

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=llm_name, temperature=llm_temperature, **kwargs)


def llama_index_anthropic_llm(
    llm_name: str = Defaults.DEFAULT_LLM_NAME,
    llm_temperature: float = Defaults.DEFAULT_LLM_TEMPERATURE,
    **kwargs,
):
    """Initialize an Anthropic LLM client for use with LlamaIndex.

    Args:
        llm_name: Name of the LLM in Anthropic API.
        llm_temperature: Temperature for the LLM.
    """
    ensure_module_is_installed("llama_index")
    from llama_index.llms.anthropic import Anthropic

    return Anthropic(model=llm_name, temperature=llm_temperature, **kwargs)


Defaults.LLM_MAP = {
    (LLMFramework.LANGCHAIN, LLMFamily.OPENAI): langchain_openai_llm,
    (LLMFramework.LLAMA_INDEX, LLMFamily.OPENAI): llama_index_openai_llm,
    (LLMFramework.LANGCHAIN, LLMFamily.ANTHROPIC): langchain_anthropic_llm,
    (LLMFramework.LLAMA_INDEX, LLMFamily.ANTHROPIC): llama_index_anthropic_llm,
}


def init_llm(
    llm_framework: str,
    llm_family: str = Defaults.DEFAULT_LLM_FAMILY,
    llm_name: str = Defaults.DEFAULT_LLM_NAME,
    llm_temperature: float = Defaults.DEFAULT_LLM_TEMPERATURE,
    **kwargs,
):
    """Initialize an LLM client for use with the specified framework and family.

    Args:
        llm_framework: Framework of the LLM client.
        llm_family: Family of the LLM.
        llm_name: Name of the LLM.
        llm_temperature: Temperature for the LLM.
    """

    func = Defaults.LLM_MAP.get((llm_framework, llm_family), None)
    if func is not None:
        return func(llm_name=llm_name, llm_temperature=llm_temperature, **kwargs)

    raise LLMFamilyNotSupported(llm_framework=llm_framework, llm_family=llm_family)
