from typing import List, Optional

import pytest

import kuzu
from langchain_core.runnables import Runnable

from motleycrew import MotleyCrew, MotleyTool
from motleycrew.tasks import TaskRecipe, TaskType
from motleycrew.storage import MotleyKuzuGraphStore
from motleycrew.common.exceptions import TaskDependencyCycleError


class TaskRecipeMock(TaskRecipe):
    def identify_candidates(self) -> List[TaskType]:
        pass

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        pass


def create_dummy_task_recipe(crew: MotleyCrew, name: str):
    return TaskRecipeMock(
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
def task_recipe_1(crew):
    return create_dummy_task_recipe(crew, "1")


@pytest.fixture
def task_recipe_2(crew):
    return create_dummy_task_recipe(crew, "2")


@pytest.fixture
def task_recipe_3(crew):
    return create_dummy_task_recipe(crew, "3")


class TestSetUpstream:
    def test_set_upstream_returns_self(self, task_recipe_1, task_recipe_2):
        result = task_recipe_2.set_upstream(task_recipe_1)

        assert result is task_recipe_2

    def test_set_upstream_sets_upstream(self, task_recipe_1, task_recipe_2):
        task_recipe_2.set_upstream(task_recipe_1)

        assert task_recipe_1.get_upstream_task_recipes() == []
        assert task_recipe_2.get_upstream_task_recipes() == [task_recipe_1]

    def test_set_upstream_sets_downstreams(self, task_recipe_1, task_recipe_2):
        task_recipe_2.set_upstream(task_recipe_1)

        assert task_recipe_1.get_downstream_task_recipes() == [task_recipe_2]
        assert task_recipe_2.get_downstream_task_recipes() == []

    def test_rshift_returns_left(self, task_recipe_1, task_recipe_2):
        result = task_recipe_1 >> task_recipe_2

        assert result is task_recipe_1

    def test_rshift_sets_downstream(self, task_recipe_1, task_recipe_2):
        task_recipe_1 >> task_recipe_2

        assert task_recipe_1.get_downstream_task_recipes() == [task_recipe_2]
        assert task_recipe_2.get_downstream_task_recipes() == []

    def test_rshift_sets_upstream(self, task_recipe_1, task_recipe_2):
        task_recipe_1 >> task_recipe_2

        assert task_recipe_1.get_upstream_task_recipes() == []
        assert task_recipe_2.get_upstream_task_recipes() == [task_recipe_1]

    def test_rshift_set_multiple_downstream(self, task_recipe_1, task_recipe_2, task_recipe_3):
        task_recipe_1 >> [task_recipe_2, task_recipe_3]

        assert set(task_recipe_1.get_downstream_task_recipes()) == {task_recipe_2, task_recipe_3}
        assert task_recipe_2.get_downstream_task_recipes() == []
        assert task_recipe_3.get_downstream_task_recipes() == []

    def test_rshift_set_multiple_upstream(self, task_recipe_1, task_recipe_2, task_recipe_3):
        task_recipe_1 >> [task_recipe_2, task_recipe_3]

        assert task_recipe_1.get_upstream_task_recipes() == []
        assert task_recipe_2.get_upstream_task_recipes() == [task_recipe_1]
        assert task_recipe_3.get_upstream_task_recipes() == [task_recipe_1]

    def test_sequence_on_left_returns_sequence(self, task_recipe_1, task_recipe_2, task_recipe_3):
        result = [task_recipe_1, task_recipe_2] >> task_recipe_3

        assert result == [task_recipe_1, task_recipe_2]

    def test_sequence_on_left_sets_downstream(self, task_recipe_1, task_recipe_2, task_recipe_3):
        [task_recipe_1, task_recipe_2] >> task_recipe_3

        assert task_recipe_1.get_downstream_task_recipes() == [task_recipe_3]
        assert task_recipe_2.get_downstream_task_recipes() == [task_recipe_3]
        assert task_recipe_3.get_downstream_task_recipes() == []

    def test_sequence_on_left_sets_upstream(self, task_recipe_1, task_recipe_2, task_recipe_3):
        [task_recipe_1, task_recipe_2] >> task_recipe_3

        assert task_recipe_1.get_upstream_task_recipes() == []
        assert task_recipe_2.get_upstream_task_recipes() == []
        assert set(task_recipe_3.get_upstream_task_recipes()) == {task_recipe_1, task_recipe_2}

    def test_deduplicates(self, task_recipe_1, task_recipe_2):
        task_recipe_1 >> [task_recipe_2, task_recipe_2]

        assert task_recipe_1.get_downstream_task_recipes() == [task_recipe_2]

    def test_error_on_direct_dependency_cycle(self, task_recipe_1):
        with pytest.raises(TaskDependencyCycleError):
            task_recipe_1 >> task_recipe_1
