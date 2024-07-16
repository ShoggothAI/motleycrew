import pytest

from motleycrew.tasks.simple import SimpleTask, SimpleTaskUnit
from tests.test_crew import CrewFixtures


class TestCrew(CrewFixtures):

    def test_create_simple_task(self, crew, agent):
        assert len(crew.tasks) == 0
        simple_task = SimpleTask(crew=crew, description="task description", agent=agent)
        assert len(crew.tasks) == 1
        node = simple_task.node
        assert crew.graph_store.get_node_by_class_and_id(type(node), node.id) == node

    @pytest.mark.parametrize("tasks", [2], indirect=True)
    def test_add_dependency(self, crew, tasks):
        task1, task2 = tasks
        crew.add_dependency(task1, task2)
        assert crew.graph_store.check_relation_exists(task1.node, task2.node)

    @pytest.mark.parametrize("tasks", [1], indirect=True)
    def test_register_added_task(self, tasks, crew):
        task = tasks[0]
        len_tasks = len(crew.tasks)
        crew.register_tasks([task])
        assert len(crew.tasks) == len_tasks

    def test_get_available_task(self, crew):
        tasks = crew.get_available_tasks()
        assert len(tasks) == 3

    def test_get_extra_tools(self, crew):
        tasks = crew.get_available_tasks()
        assert not crew.get_extra_tools(tasks[-1])

    def test_run(self, crew, agent):
        available_tasks = crew.get_available_tasks()
        crew.run()
        for task in crew.tasks:
            assert task.done
            assert task.node.done
            unit = SimpleTaskUnit(
                name=task.name,
                prompt=task.description,
            )
            if task in available_tasks:
                assert agent.invoke(unit.as_dict()) == task.output
            else:
                assert agent.invoke(unit.as_dict()) != task.output
