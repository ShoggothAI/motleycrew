import pytest
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI

from motleycrew.common import LLMProvider, LLMFramework
from motleycrew.common.exceptions import LLMProviderNotSupported
from motleycrew.common.llms import init_llm


@pytest.mark.parametrize(
    "llm_provider, llm_framework, expected_class",
    [
        (LLMProvider.OPENAI, LLMFramework.LANGCHAIN, ChatOpenAI),
        (LLMProvider.OPENAI, LLMFramework.LLAMA_INDEX, OpenAI),
    ],
)
def test_init_llm(llm_provider, llm_framework, expected_class):
    llm = init_llm(llm_provider=llm_provider, llm_framework=llm_framework)
    assert isinstance(llm, expected_class)


def test_raise_init_llm():
    with pytest.raises(LLMProviderNotSupported):
        llm = init_llm(llm_provider=LLMProvider.OPENAI, llm_framework="unknown_framework")
