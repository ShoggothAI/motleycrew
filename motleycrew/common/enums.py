class LLMFamily:
    """ Description

    Attributes:
        OPENAI (str):

    """
    OPENAI = "openai"


class LLMFramework:
    """ Description

    Attributes:
        LANGCHAIN (str):
        LLAMA_INDEX (str):

    """
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"


class GraphStoreType:
    """ Description

    Attributes:
        KUZU (str):

    """
    KUZU = "kuzu"


class TaskUnitStatus:
    """Description

    Attributes:
        PENDING (str):
        RUNNING (str):
        DONE (str):
    """
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


class LunaryRunType:
    """ Description

    Attributes:
        LLM (str):
        AGENT (str):
        TOOL (str):
        CHAIN (str):
        EMBED (str):

    """
    LLM = "llm"
    AGENT = "agent"
    TOOL = "tool"
    CHAIN = "chain"
    EMBED = "embed"


class LunaryEventName:
    """ Description

    Attributes:
        START (str):
        END (str):
        UPDATE (str):
        ERROR (str):

    """
    START = "start"
    END = "end"
    UPDATE = "update"
    ERROR = "error"
