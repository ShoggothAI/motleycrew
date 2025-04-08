from typing import List, Optional, Any, Callable, Type

from pydantic import BaseModel, Field

from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.tools import MotleyTool
from motleycrew.tools.structured_passthrough import StructuredPassthroughTool


class PivotConfigToolInputSchema(BaseModel):
    question: str = Field(description="The question to answer with the pivot chart.")
    datasource_kv_store_keys: List[str] = Field(
        description="The key(s) of the datasource(s) to use in the KV store."
    )


class AgenticValidationLoop(MotleyTool):

    def __init__(
        self,
        schema: Type[BaseModel],
        name: str,
        description: str,
        post_process: Optional[Callable] = None,
        llm: Optional[Any] = None,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.llm = llm or init_llm(LLMFramework.LANGCHAIN)
        self.schema = schema
        self.post_process = post_process

    def run(self, prompt: str) -> Any:
        """
        Run the tool with the provided inputs.
        """
        output_tool = StructuredPassthroughTool(
            schema=self.schema,
            post_process=self.post_process,
            exceptions_to_reflect=[Exception],
        )

        agent = ReActToolCallingMotleyAgent(
            tools=[output_tool],
            llm=self.llm,
            name=self.name + "_agent",
            force_output_handler=True,
            prompt_prefix=prompt,
        )

        # Run the agent with the prompt
        response = agent.invoke({})

        return response
