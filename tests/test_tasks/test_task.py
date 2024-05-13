import pytest

from motleycrew import MotleyCrew
from motleycrew.tasks import TaskRecipe
from motleycrew.common.exceptions import TaskDependencyCycleError


def create_dummy_task(crew: MotleyCrew, name: str):
    return TaskRecipe(
        name=name,
        description=name,
        crew=crew,
    )


@pytest.fixture
def crew():
    return MotleyCrew()


@pytest.fixture
def task1(crew):
    return create_dummy_task(crew, "1")


@pytest.fixture
def task2(crew):
    return create_dummy_task(crew, "2")


@pytest.fixture
def task3(crew):
    return create_dummy_task(crew, "3")


class TestSetUpstream:
    def test_set_upstream_returns_self(self, task1, task2):
        result = task2.set_upstream(task1)

        assert result is task2

    def test_set_upstream_sets_upstream(self, task1, task2):
        task2.set_upstream(task1)

        assert task1.upstream_tasks == []
        assert task2.upstream_tasks == [task1]

    def test_set_upstream_sets_downstreams(self, task1, task2):
        task2.set_upstream(task1)

        assert task1.downstream_tasks == [task2]
        assert task2.downstream_tasks == []

    def test_rshift_returns_left(self, task1, task2):
        result = task1 >> task2

        assert result is task1

    def test_rshift_sets_downstream(self, task1, task2):
        task1 >> task2

        assert task1.downstream_tasks == [task2]
        assert task2.downstream_tasks == []

    def test_rshift_sets_upstream(self, task1, task2):
        task1 >> task2

        assert task1.upstream_tasks == []
        assert task2.upstream_tasks == [task1]

    def test_rshift_set_multiple_downstream(self, task1, task2, task3):
        task1 >> [task2, task3]

        assert task1.downstream_tasks == [task2, task3]
        assert task2.downstream_tasks == []
        assert task3.downstream_tasks == []

    def test_rshift_set_multiple_upstream(self, task1, task2, task3):
        task1 >> [task2, task3]

        assert task1.upstream_tasks == []
        assert task2.upstream_tasks == [task1]
        assert task3.upstream_tasks == [task1]

    def test_sequence_on_left_returns_sequence(self, task1, task2, task3):
        result = [task1, task2] >> task3

        assert result == [task1, task2]

    def test_sequence_on_left_sets_downstream(self, task1, task2, task3):
        [task1, task2] >> task3

        assert task1.downstream_tasks == [task3]
        assert task2.downstream_tasks == [task3]
        assert task3.downstream_tasks == []

    def test_sequence_on_left_sets_upstream(self, task1, task2, task3):
        [task1, task2] >> task3

        assert task1.upstream_tasks == []
        assert task2.upstream_tasks == []
        assert task3.upstream_tasks == [task1, task2]

    def test_deduplicates(self, task1, task2):
        task1 >> [task2, task2]

        assert task1.downstream_tasks == [task2]

    def test_error_on_direct_dependency_cycle(self, task1):
        with pytest.raises(TaskDependencyCycleError):
            task1 >> task1
