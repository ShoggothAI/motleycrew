"""Various enums used in the project."""


class LLMFamily:
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMFramework:
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"


class GraphStoreType:
    KUZU = "kuzu"


class TaskUnitStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


class LunaryRunType:
    LLM = "llm"
    AGENT = "agent"
    TOOL = "tool"
    CHAIN = "chain"
    EMBED = "embed"


class LunaryEventName:
    START = "start"
    END = "end"
    UPDATE = "update"
    ERROR = "error"


class AsyncBackend:
    """Backends for parallel crew execution.

    Attributes:
        ASYNCIO: Asynchronous execution using asyncio.
        THREADING: Parallel execution using threads.
        NONE: Synchronous execution.
    """

    ASYNCIO = "asyncio"
    THREADING = "threading"
    NONE = "none"
