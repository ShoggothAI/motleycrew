class LLMFamilyNotSupported(Exception):
    def __init__(self, llm_framework: str, llm_family: str):
        self.llm_framework = llm_framework
        self.llm_family = llm_family

    def __str__(self) -> str:
        return f"LLM family `{self.llm_family}` is not supported via the framework `{self.llm_framework}`"


class LLMFrameworkNotSupported(Exception):
    def __init__(self, llm_framework: str):
        self.llm_framework = llm_framework

    def __str__(self) -> str:
        return f"LLM framework `{self.llm_framework}` is not supported"


class AgentNotMaterialized(Exception):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def __str__(self) -> str:
        return f"Agent `{self.agent_name}` is not yet materialized"


class CannotModifyMaterializedAgent(Exception):
    def __init__(self, agent_name: str | None):
        self.agent_name = agent_name

    def __str__(self) -> str:
        return "Cannot modify agent{} as it is already materialized".format(
            f" `{self.agent_name}`" if self.agent_name is not None else ""
        )
