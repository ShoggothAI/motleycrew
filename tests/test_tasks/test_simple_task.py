import pytest

from langchain_community.tools import DuckDuckGoSearchRun
from motleycrew.crew import MotleyCrew
from motleycrew.agents.langchain.openai_tools_react import ReactOpenAIToolsAgent
from motleycrew.tasks.simple import (
    SimpleTask,
    SimpleTaskUnit,
    compose_simple_task_prompt_with_dependencies,
)


class TestSimpleTask:
    @pytest.fixture(scope="class")
    def crew(self):
        obj = MotleyCrew()
        return obj

    @pytest.fixture(scope="class")
    def agent(self):
        agent = ReactOpenAIToolsAgent(
            name="AI writer agent",
            tools=[DuckDuckGoSearchRun()],
            verbose=True,
        )
        return agent

    @pytest.fixture(scope="class")
    def tasks(self, crew, agent):
        task1 = SimpleTask(crew=crew, description="task1 description", agent=agent)
        task2 = SimpleTask(crew=crew, description="task2 description")
        crew.register_tasks([task1, task2])
        return [task1, task2]

    def test_register_completed_unit(self, tasks, crew):
        task1, task2 = tasks
        assert not task1.done
        assert task1.output is None
        unit = task1.get_next_unit()
        unit.output = task1.description

        with pytest.raises(AssertionError):
            task1.register_completed_unit(unit)
        unit.set_done()
        task1.register_completed_unit(unit)
        assert task1.done
        assert task1.output == unit.output
        assert task1.node.done

    def test_get_next_unit(self, tasks, crew):
        task1, task2 = tasks
        crew.add_dependency(task1, task2)
        assert task1.get_next_unit() is None
        prompt = compose_simple_task_prompt_with_dependencies(task2.description, task2.get_units())
        expected_unit = SimpleTaskUnit(
            name=task2.name,
            prompt=prompt,
        )
        next_unit = task2.get_next_unit()
        assert next_unit.prompt == expected_unit.prompt

    def test_get_worker(self, tasks, agent):
        task1, task2 = tasks
        assert task1.get_worker([]) == agent
        with pytest.raises(ValueError):
            task2.get_worker([])
