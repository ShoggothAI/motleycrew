Choosing LLMs
====================

Generally, the interaction with an LLM is up to the agent implementation.
However, as motleycrew integrates with several agent frameworks, there is some common ground for how to choose LLMs.


Providing an LLM to an agent
----------------------------

In general, you can pass a specific LLM to the agent you're using.

.. code-block:: python

    from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = ReActToolCallingMotleyAgent(llm=llm, tools=[...])


The LLM class depends on the agent framework you're using.
That's why we have an ``init_llm`` function to help you set up the LLM.

.. code-block:: python

    from motleycrew.common.llms import init_llm
    from motleycrew.common import LLMFramework, LLMProvider

    llm = init_llm(
        llm_framework=LLMFramework.LANGCHAIN,
        llm_provider=LLMProvider.ANTHROPIC,
        llm_name="claude-3-5-sonnet-latest",
        llm_temperature=0
    )
    agent = ReActToolCallingMotleyAgent(llm=llm, tools=[...])


The currently supported frameworks (:py:class:`motleycrew.common.enums.LLMFramework`) are:

- :py:class:`Langchain <motleycrew.common.enums.LLMFramework.LANGCHAIN>` for Langchain-based agents from Langchain, CrewAI, motelycrew etc.
- :py:class:`LlamaIndex <motleycrew.common.enums.LLMFramework.LLAMA_INDEX>` for LlamaIndex-based agents.

The currently supported LLM providers (:py:class:`motleycrew.common.enums.LLMProvider`) are:

- :py:class:`OpenAI <motleycrew.common.enums.LLMProvider.OPENAI>`
- :py:class:`Anthropic <motleycrew.common.enums.LLMProvider.ANTHROPIC>`
- :py:class:`Groq <motleycrew.common.enums.LLMProvider.GROQ>`
- :py:class:`Together <motleycrew.common.enums.LLMProvider.TOGETHER>`
- :py:class:`Replicate <motleycrew.common.enums.LLMProvider.REPLICATE>`
- :py:class:`Ollama <motleycrew.common.enums.LLMProvider.OLLAMA>`
- :py:class:`Azure OpenAI <motleycrew.common.enums.LLMProvider.AZURE_OPENAI>`

Please raise an issue if you need to add support for another LLM provider.


Default LLM
-----------

At present, we default to OpenAI's latest ``gpt-4o`` model for our agents,
and rely on the user to set the `OPENAI_API_KEY` environment variable.

You can control the default LLM as follows:

.. code-block:: python

    from motleycrew.common import Defaults
    Defaults.DEFAULT_LLM_PROVIDE = "the_new_default_LLM_provider"
    Defaults.DEFAULT_LLM_NAME = "name_of_the_new_default_model_from_the_provider"


Using custom LLMs
-----------------

To use a custom LLM provider to use as the default or via the ``init_llm`` function,
you need to make sure that for all the frameworks you're using (currently at most Langchain, LlamaIndex),
the `LLM_MAP` has an entry for the LLM provider, for example as follows:

.. code-block:: python

    from motleycrew.common import LLMProvider
    from motleycrew.common.llms import LLM_MAP

    LLM_MAP[(LLMFramework.LANGCHAIN, "MyLLMProvider")] = my_langchain_llm_factory
    LLM_MAP[(LLMFramework.LLAMA_INDEX, "MyLLMProvider")] = my_llamaindex_llm_factory

Here each llm factory is a function with a signature
``def llm_factory(llm_name: str, llm_temperature: float, **kwargs)`` that returns the model object for the relevant framework.

For example, this is the built-in OpenAI model factory for Langchain:

.. code-block:: python

    def langchain_openai_llm(
        llm_name: str = Defaults.DEFAULT_LLM_NAME,
        llm_temperature: float = Defaults.DEFAULT_LLM_TEMPERATURE,
        **kwargs,
    ):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm_name, temperature=llm_temperature, **kwargs)


You can also overwrite the `LLM_MAP` values for e.g. the OpenAI models if, for example,
you want to use an in-house wrapper for Langchain or Llamaindex model adapters
(for example, to use an internal gateway instead of directly hitting the OpenAI endpoints).

Note that at present, if you use Autogen with motleycrew, you will need to separately control
the models that Autogen uses, using the Autogen-specific APIs.
