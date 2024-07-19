import pytest
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.crew import MotleyCrew
from motleycrew.storage.graph_store_utils import init_graph_store
from motleycrew.tasks.simple import (
    SimpleTask,
    SimpleTaskUnit,
    compose_simple_task_prompt_with_dependencies,
)


@pytest.fixture(scope="session")
def graph_store():
    graph_store = init_graph_store()
    return graph_store


@pytest.fixture
def crew(graph_store):
    return MotleyCrew(graph_store=graph_store)


@pytest.fixture
def agent():
    agent = ReActToolCallingMotleyAgent(
        name="AI writer agent",
        tools=[DuckDuckGoSearchRun()],
        verbose=True,
    )
    return agent


@pytest.fixture
def tasks(crew, agent):
    task1 = SimpleTask(crew=crew, description="task1 description", agent=agent)
    task2 = SimpleTask(crew=crew, description="task2 description")
    crew.register_tasks([task1, task2])
    return [task1, task2]


class TestSimpleTask:
    def test_register_completed_unit(self, tasks, crew):
        task1, task2 = tasks
        assert not task1.done
        assert task1.output is None
        unit = task1.get_next_unit()
        unit.output = task1.description

        with pytest.raises(AssertionError):
            task1.on_unit_completion(unit)
        unit.set_done()
        task1.on_unit_completion(unit)
        assert task1.done
        assert task1.output == unit.output
        assert task1.node.done

    def test_get_next_unit(self, tasks, crew):
        task1, task2 = tasks
        crew.add_dependency(task1, task2)
        assert task2.get_next_unit() is None
        prompt = compose_simple_task_prompt_with_dependencies(
            description=task1.description,
            upstream_task_units=task1.get_units(),
            prompt_template_with_upstreams=task1.prompt_template_with_upstreams,
        )
        expected_unit = SimpleTaskUnit(
            name=task1.name,
            prompt=prompt,
        )
        next_unit = task1.get_next_unit()
        assert next_unit.prompt == expected_unit.prompt

    def test_get_worker(self, tasks, agent):
        task1, task2 = tasks
        assert task1.get_worker([]) == agent
        with pytest.raises(ValueError):
            task2.get_worker([])
