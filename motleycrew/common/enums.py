"""Various enums used in the project."""


class LLMProvider:
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    REPLICATE = "replicate"
    TOGETHER = "together"
    GROQ = "groq"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"

    ALL = {OPENAI, ANTHROPIC, REPLICATE, TOGETHER, GROQ, OLLAMA, AZURE_OPENAI}


class LLMFramework:
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"

    ALL = {LANGCHAIN, LLAMA_INDEX}


class GraphStoreType:
    KUZU = "kuzu"

    ALL = {KUZU}


class TaskUnitStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"

    ALL = {PENDING, RUNNING, DONE}


class LunaryRunType:
    LLM = "llm"
    AGENT = "agent"
    TOOL = "tool"
    CHAIN = "chain"
    EMBED = "embed"

    ALL = {LLM, AGENT, TOOL, CHAIN, EMBED}


class LunaryEventName:
    START = "start"
    END = "end"
    UPDATE = "update"
    ERROR = "error"

    ALL = {START, END, UPDATE, ERROR}


class AsyncBackend:
    """Backends for parallel crew execution.

    Attributes:
        ASYNCIO: Asynchronous execution using asyncio.
        THREADING: Parallel execution using threads.
        RAY: Parallel execution using Ray.
        NONE: Synchronous execution.
    """

    ASYNCIO = "asyncio"
    THREADING = "threading"
    RAY = "ray"
    NONE = "none"

    ALL = {ASYNCIO, THREADING, RAY, NONE}
