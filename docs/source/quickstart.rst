Quickstart
==========

This is a brief introduction to motleycrew.

For a working example of agents, tools, crew, and SimpleTask, check out the :doc:`blog with images <examples/blog_with_images>`.

For a working example of custom tasks that fully utilize the knowledge graph backend, check out the :doc:`research agent <examples/research_agent>`.

Agents and tools
----------------

Motleycrew provides thin wrappers for all the common agent frameworks: Langchain, LlamaIndex, CrewAI, and Autogen (please let us know if you want any others added!).
It also provides thin wrappers for Langchain and LlamaIndex tools, allowing you to use any of these tools with any of these agents.

MotleyCrew also supports **delegation**: you can simply give any agent as a tool to any other agent.

All the wrappers for tools and agents implement the Runnable interface, so you can use them as-is in LCEL and Langgraph code.

Output handlers (aka return_direct)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An **output handler** is a tool that the agent uses for submitting the final output instead of returning it raw. Besides defining a schema for the output, the output handler enables you to implement any validation logic inside it, including agent-based. If your agent returns invalid output, you can raise an exception that will be returned to the agent so that it can retry.

Essentially, an output handler is a tool that returns its output directly to the user, thus finishing the agent execution. This behavior is enabled by setting the ``return_direct=True`` argument for the tool. Unlike other frameworks, MotleyCrew allows to have multiple output handlers for one agent, from which the agent can choose one.

MotleyCrew also supports **forced output handlers**. This means that the agent will only be able to return output via an output handler, and not directly. This is useful if you want to ensure that the agent only returns output in a specific format.

See our usage examples with a :doc:`simple validator <examples/validating_agent_output>` and an :doc:`advanced output handler with multiple fields <examples/advanced_output_handling>`.

MotleyTool
^^^^^^^^^^

A tool in motleycrew, like in other frameworks, is basically a function that takes an input and returns an output.
It is called a tool in the sense that it is usually used by an agent to perform a specific action.
Besides the function itself, a tool also contains an input schema which describes the input format to the LLM.

``MotleyTool`` is the base class for all tools in motleycrew. It is a subclass of ``Runnable`` that adds some additional features to the tool, along with necessary adapters and converters.

If you pass a tool from a supported framework (currently Langchain, LlamaIndex, and CrewAI) to a motleycrew agent, it will be automatically converted. If you want to have control over this, e.g. to customize tool params, you can do it manually.

.. code-block:: python

    motley_tool = MotleyTool.from_supported_tool(my_tool)

It is also possible to define a custom tool using the ``MotleyTool`` base class, overriding the ``run`` method. This is especially useful if you want to access context such as the caller agent or its last input, which can be useful for validation.

.. code-block:: python

    class MyTool(MotleyTool):
        def run(self, some_input: str) -> str:
            return f"Received {some_input} from agent {self.agent} with last input {self.agent_input}"

Tools can be executed asynchronously, either directly of via by an asynchronous agent. By default, the async version will just run the sync version in a separate thread

MotleyTool can reflect exceptions that are raised inside it back to the agent, which can then retry the tool call. You can pass a list of exception classes to the ``exceptions_to_reflect`` argument in the constructor (or even pass the ``Exception`` class to reflect everything).

Crew and tasks
--------------

The other two key concepts in motleycrew are crew and tasks. The crew is the orchestrator for tasks, and must be passed to all tasks at creation; tasks can be connected into a DAG using the ``>>`` operator, for example ``TaskA >> TaskB``. This means that ``TaskB`` will not be started before ``TaskA`` is complete, and will be given ``TaskA``'s output.

Once all tasks and their relationships have been set up, it all can be run via ``crew.run()``, which returns a list of the executed ``TaskUnits`` (see below for details).

SimpleTask
^^^^^^^^^^

``SimpleTask`` is a basic implementation of the ``Task`` API. It only requires a crew, a description, and an agent. When it's executed, the description is combined with the output of any upstream tasks and passed on to the agent, and the agent's output is the tasks's output.

For a working illustration of all the concepts so far, see the :doc:`blog with images <examples/blog_with_images>` example.

Knowledge graph backend and custom tasks
----------------------------------------

The functionality so far is convenient, allowing us to mix all the popular agents and tools, but otherwise fairly vanilla, little different from, for example, the CrewAI semantics. Fortunately, the above introduction just scratched the surface of the motleycrew ``Task`` API.

In motleycrew, a task is basically a set of rules describing how to perform actions. It provides a **worker** (e.g. an agent) and sets of input data called **task units**. This allows defining workflows of any complexity concisely using crew semantics. For a deeper dive, check out the page on :doc:`key concepts <key_concepts>`.

The crew queries and dispatches available task units in a loop, managing task states using an embedded :doc:`knowledge graph <knowledge_graph>`.

This dispatch method easily supports different execution backends, from synchronous to asyncio, threaded, etc.

Example: Recursive question-answering in the research agent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Motleycrew architecture described above easily allows to generate task units on the fly, if needed. An example of the power of this approach is the :doc:`research agent <examples/research_agent>` that dynamically generates new questions based on retrieved context for previous questions.
This example also shows how workers can collaborate via the shared knowledge graph, storing all necessary data in a way that is natural to the task.
