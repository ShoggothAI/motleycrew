from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Set

from motleycrew.agent.parent import MotleyAgentAbstractParent

if TYPE_CHECKING:
    from motleycrew import MotleyCrew

PROMPT_TEMPLATE_WITH_DEPS = """
{description}

You must use the results of these upstream tasks:

{upstream_results_section}
"""


class TaskDependencyCycleError(Exception):
    """Raised when a task is set to depend on itself"""


class Task:
    def __init__(
        self,
        description: str,
        name: str,
        crew: MotleyCrew,
        agent: MotleyAgentAbstractParent | None = None,
        documents: Sequence[Any] | None = None,
        creator_name: str | None = None,
        return_to_creator: bool = False,
    ):
        self.name = name  # does it really need one? Where does it get used?
        self.description = description
        self.crew = crew
        self.agent = agent  # to be auto-assigned at crew creation if missing?
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

        self.upstream_tasks: Set[Task] = set()
        self.downstream_tasks: Set[Task] = set()

        self.crew.add_task(self)

    def prompt(self) -> str:
        """
        For compatibility with crewai.Agent.execute_task
        :return:
        """
        if not self.upstream_tasks:
            return self.description

        # TODO include the rest of the outputs list
        upstream_results = [f"##{t.name}\n{t.outputs[-1]}" for t in self.upstream_tasks]
        upstream_results_section = "\n\n".join(upstream_results)
        return PROMPT_TEMPLATE_WITH_DEPS.format(
            description=self.description,
            upstream_results_section=upstream_results_section,
        )

    def increment_tools_errors(self) -> None:
        """
        For compatibility with crewai.Agent.execute_task
        It is called when an exception is raised in the tool, so for now we just re-raise it here
        TODO: do we want to handle tool errors in future like CrewAI handles them?
        :return:
        """
        raise

    def is_ready(self) -> bool:
        return not self.done and all(t.done for t in self.upstream_tasks)

    def set_upstream(self, task: Task) -> Task:
        if task is self:
            raise TaskDependencyCycleError(f"Task {task.name} can not depend on itself")

        self.upstream_tasks.add(task)
        task.downstream_tasks.add(self)

        return self

    def __rshift__(self, other: Task | Sequence[Task]) -> Task:
        if isinstance(other, Task):
            tasks = {other}
        else:
            tasks = other

        for task in tasks:
            task.set_upstream(self)

        return self

    def __rrshift__(self, other: Sequence[Task]) -> Sequence[Task]:
        for task in other:
            self.set_upstream(task)
        return other
