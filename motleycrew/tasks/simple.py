from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, List, Optional
from uuid import uuid4

from motleycrew.tasks.parent import Task, TaskRecipe, TaskRecipeNode

try:
    from crewai import Task as CrewAITask
except ImportError:
    CrewAITask = None

from motleycrew.agent.parent import MotleyAgentAbstractParent
from motleycrew.tool import MotleyTool

if TYPE_CHECKING:
    pass

PROMPT_TEMPLATE_WITH_DEPS = """
{description}

You must use the results of these upstream tasks:

{upstream_results_section}
"""


class SimpleTask(Task):
    def __init__(self, node_id: int, status: str = "pending", outputs: Optional[List[Any]] = None):
        super().__init__(status, outputs)
        self.node_id = node_id

    def to_crewai_task(self) -> CrewAITask:
        if CrewAITask is None:
            raise ImportError("crewai is not installed")

        raise NotImplementedError("This method is not yet implemented")
        # return CrewAITask(
        #     name=self.name or self.description, description=self.description, prompt=self.prompt
        # )


@dataclass
class SimpleTaskRecipeNode(TaskRecipeNode):
    pass


class SimpleTaskRecipe(TaskRecipe):

    def __init__(
        self,
        description: str,
        name: str | None = None,
        agent: MotleyAgentAbstractParent | None = None,
        tools: Sequence[MotleyTool] | None = None,
        documents: Sequence[Any] | None = None,
        creator_name: str | None = None,
        return_to_creator: bool = False,
    ):
        super().__init__(name)
        self.description = description
        self.agent = agent  # to be auto-assigned at crew creation if missing?
        self.tools = tools or []
        # should tasks own agents or should agents own tasks?
        self.documents = documents  # to be passed to an auto-init'd retrieval, later on
        self.creator_name = creator_name or "Human"
        self.return_to_creator = (
            return_to_creator  # for orchestrator to know to send back to creator
        )
        self.message_history = []  # Useful when task is passed around between agents
        self.outputs = []  # to be filled in by the agent(s) once the task is complete
        self.used_tools = 0  # a hack for CrewAI compatibility

        self.done: bool = False
        self.id = str(uuid4())

        # This will be set by MotleyCrew.register_task
        self.crew = None

    def summary(self) -> SimpleTaskRecipeNode:
        # TODO: fix
        return SimpleTaskRecipeNode(
            id=self.id,
            name=self.name,
            description=self.description,
            done=self.done,
            outputs=self.outputs,
        )

    def identify_candidates(self) -> List[SimpleTask]:
        # TODO: implement via graph
        return not self.done and all(t.done for t in self.upstream_tasks)

    def get_agent(self) -> MotleyAgentAbstractParent:
        # TODO: implement this via agent factory
        if self.agent is None:
            raise ValueError("Task is not associated with an agent")
        return self.agent
