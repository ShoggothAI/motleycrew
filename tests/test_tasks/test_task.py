from typing import List, Optional

import pytest

import kuzu
from langchain_core.runnables import Runnable

from motleycrew import MotleyCrew
from motleycrew.tools import MotleyTool
from motleycrew.tasks import Task, TaskUnitType, TaskUnit
from motleycrew.storage import MotleyKuzuGraphStore
from motleycrew.common.exceptions import TaskDependencyCycleError


class TaskMock(Task):
    def get_next_unit(self) -> List[TaskUnitType]:
        pass

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        pass


def create_dummy_task(crew: MotleyCrew, name: str):
    return TaskMock(
        name=name,
        crew=crew,
    )


@pytest.fixture
def graph_store(tmpdir):
    db_path = tmpdir / "test_db"
    db = kuzu.Database(str(db_path))
    graph_store = MotleyKuzuGraphStore(db)
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

    @pytest.fixture(scope="class")
    def task(self):
        return create_dummy_task(MotleyCrew(), "test task")

    def test_register_started_unit(self, task):
        with pytest.raises(AssertionError):
            task.register_started_unit("unit")

        unit = TaskUnit()
        unit.set_done()

        with pytest.raises(AssertionError):
            task.register_started_unit(unit)

    def test_set_done(self, task):
        assert not task.done
        assert not task.node.done
        task.set_done()
        assert task.done
        assert task.node.done
