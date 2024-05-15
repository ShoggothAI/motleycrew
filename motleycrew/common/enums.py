class LLMFamily:
    OPENAI = "openai"


class LLMFramework:
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"


class GraphStoreType:
    KUZU = "kuzu"


class TaskStatus:
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
    ERROR = "error"
