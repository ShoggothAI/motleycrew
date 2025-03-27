Key-value store for manipulating objects
========================================

In many real-world tasks, the agent needs to deal with different types of data.
Even assuming that the LLM can reliably manipulate text, you often need to work
with other objects, such as structured data in the form of dataframes, JSONs,
or just any Python objects.

MotleyCrew provides a simple way to store and retrieve objects using a key-value store,
allowing the agent to use them in its operations, like passing them to tools
(e.g. calling a query tool to get some statistics about a particular dataset).

The key-value store is a dictionary that can be accessed at ``agent.kv_store``.


.. code-block:: python
    class ObjectWriterTool(MotleyTool):
        def run(self, key: str, value: Any):
            """Write an object to the key-value store."""
            self.agent.kv_store[key] = value


    class ObjectReaderTool(MotleyTool):
        def run(self, key: str) -> Any:
            """Read an object from the key-value store."""
            return self.agent.kv_store[key]


A simple example of using the key-value store can be found
`here <https://github.com/ShoggothAI/motleycrew/blob/main/examples/key_value_store.py>`_.
