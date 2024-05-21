import pytest

from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI

from motleycrew.common.llms import init_llm
from motleycrew.common import LLMFamily, LLMFramework
from motleycrew.common.exceptions import LLMFamilyNotSupported


@pytest.mark.parametrize(
    "llm_family, llm_framework, expected_class",
    [
        (LLMFamily.OPENAI, LLMFramework.LANGCHAIN, ChatOpenAI),
        (LLMFamily.OPENAI, LLMFramework.LLAMA_INDEX, OpenAI),
    ],
)
def test_init_llm(llm_family, llm_framework, expected_class):
    llm = init_llm(llm_family=llm_family, llm_framework=llm_framework)
    assert isinstance(llm, expected_class)


def test_raise_init_llm():
    with pytest.raises(LLMFamilyNotSupported):
        llm = init_llm(llm_family=LLMFamily.OPENAI, llm_framework="unknown_framework")
