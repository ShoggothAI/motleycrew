Agents
======

MotleyCrew is first and foremost a multi-agent framework, so the concept of an agent is central to it.

An agent is essentially an actor that can perform tasks. Usually, it contains some kind of loop
that interacts with an LLM and performs actions based on the data it receives.


ReAct tool calling agent
------------------------
MotleyCrew provides a robust general-purpose agent that implements
`ReAct prompting <https://react-lm.github.io/>`_: :class:`motleycrew.agents.langchain.ReActToolCallingMotleyAgent`.
This agent is probably a good starting point for most tasks.

.. code-block:: python

    from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
    from langchain_community.tools import DuckDuckGoSearchRun

    agent = ReActToolCallingMotleyAgent(tools=[DuckDuckGoSearchRun()])
    agent.invoke({"prompt": "Which country currently has more population, China or India?"})


``ReActToolCallingMotleyAgent`` was tested with the newer OpenAI and Anthropic models, and it should work
with any model that supports function calling. If you want a similar agent for models without
function calling support, look at :class:`motleycrew.agents.langchain.LegacyReActMotleyAgent`
or :class:`motleycrew.agents.llama_index.ReActLlamaIndexMotleyAgent`.


Using agents from other frameworks
----------------------------------
For many tasks, it's reasonable to use a pre-built agent from some framework,
like Langchain, LlamaIndex, CrewAI etc. MotleyCrew provides adapters for these frameworks,
which allow you to mix and match different agents together and easily provide them with any tools.


* :class:`motleycrew.agents.langchain.LangchainMotleyAgent`
* :class:`motleycrew.agents.llama_index.LlamaIndexMotleyAgent`
* :class:`motleycrew.agents.crewai.CrewAIMotleyAgent`


Creating your own agent
-----------------------
The simplest way to create your own agent is to subclass :class:`motleycrew.agents.parent.MotleyAgentParent`.

Note that in a `crew <key_concepts.html#crew-and-knowledge-graph>`_,
not only an agent can be a `worker <key_concepts.html#tasks-task-units-and-workers>`_.
A worker is basically any `Runnable <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_,
and all agents and tools implement the Runnable interface in motleycrew.
