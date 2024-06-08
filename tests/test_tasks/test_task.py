from typing import List, Optional

import pytest

from langchain_core.runnables import Runnable

from motleycrew import MotleyCrew
from motleycrew.tools import MotleyTool
from motleycrew.tasks import Task, TaskUnitType, TaskUnit
from motleycrew.storage.graph_store_utils import init_graph_store
from motleycrew.common.exceptions import TaskDependencyCycleError


class TaskMock(Task):
    def get_next_unit(self) -> List[TaskUnitType]:
        pass

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        pass


def create_dummy_task(crew: MotleyCrew, name: str):
    return TaskMock(
        name=name,
        task_unit_class=TaskUnit,
        crew=crew,
    )


@pytest.fixture(scope="session")
def graph_store():
    graph_store = init_graph_store()
    return graph_store


@pytest.fixture
def crew(graph_store):
    return MotleyCrew(graph_store=graph_store)


@pytest.fixture
def task_1(crew):
    return create_dummy_task(crew, "1")


@pytest.fixture
def task_2(crew):
    return create_dummy_task(crew, "2")


@pytest.fixture
def task_3(crew):
    return create_dummy_task(crew, "3")


class TestSetUpstream:
    def test_set_upstream_returns_self(self, task_1, task_2):
        result = task_2.set_upstream(task_1)

        assert result is task_2

    def test_set_upstream_sets_upstream(self, task_1, task_2):
        task_2.set_upstream(task_1)

        assert task_1.get_upstream_tasks() == []
        assert task_2.get_upstream_tasks() == [task_1]

    def test_set_upstream_sets_downstreams(self, task_1, task_2):
        task_2.set_upstream(task_1)

        assert task_1.get_downstream_tasks() == [task_2]
        assert task_2.get_downstream_tasks() == []

    def test_rshift_returns_left(self, task_1, task_2):
        result = task_1 >> task_2

        assert result is task_1

    def test_rshift_sets_downstream(self, task_1, task_2):
        task_1 >> task_2

        assert task_1.get_downstream_tasks() == [task_2]
        assert task_2.get_downstream_tasks() == []

    def test_rshift_sets_upstream(self, task_1, task_2):
        task_1 >> task_2

        assert task_1.get_upstream_tasks() == []
        assert task_2.get_upstream_tasks() == [task_1]

    def test_rshift_set_multiple_downstream(self, task_1, task_2, task_3):
        task_1 >> [task_2, task_3]

        assert set(task_1.get_downstream_tasks()) == {task_2, task_3}
        assert task_2.get_downstream_tasks() == []
        assert task_3.get_downstream_tasks() == []

    def test_rshift_set_multiple_upstream(self, task_1, task_2, task_3):
        task_1 >> [task_2, task_3]

        assert task_1.get_upstream_tasks() == []
        assert task_2.get_upstream_tasks() == [task_1]
        assert task_3.get_upstream_tasks() == [task_1]

    def test_sequence_on_left_returns_sequence(self, task_1, task_2, task_3):
        result = [task_1, task_2] >> task_3

        assert result == [task_1, task_2]

    def test_sequence_on_left_sets_downstream(self, task_1, task_2, task_3):
        [task_1, task_2] >> task_3

        assert task_1.get_downstream_tasks() == [task_3]
        assert task_2.get_downstream_tasks() == [task_3]
        assert task_3.get_downstream_tasks() == []

    def test_sequence_on_left_sets_upstream(self, task_1, task_2, task_3):
        [task_1, task_2] >> task_3

        assert task_1.get_upstream_tasks() == []
        assert task_2.get_upstream_tasks() == []
        assert set(task_3.get_upstream_tasks()) == {task_1, task_2}

    def test_deduplicates(self, task_1, task_2):
        task_1 >> [task_2, task_2]

        assert task_1.get_downstream_tasks() == [task_2]

    def test_error_on_direct_dependency_cycle(self, task_1):
        with pytest.raises(TaskDependencyCycleError):
            task_1 >> task_1


class TestTask:
    def test_set_done(self, task_1):
        assert not task_1.done
        assert not task_1.node.done
        task_1.set_done()
        assert task_1.done
        assert task_1.node.done
