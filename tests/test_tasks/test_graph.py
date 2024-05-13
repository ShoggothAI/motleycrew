import pytest

from motleycrew import MotleyCrew, TaskRecipe


@pytest.fixture
def crew():
    return MotleyCrew()


@pytest.fixture
def graph(crew):
    return crew.task_graph


def test_empty_graph_has_no_tasks_remaining(graph):
    assert graph.num_tasks_remaining() == 0


def test_registered_tasks_are_counted_as_pending(crew, graph):
    _ = TaskRecipe("", "", crew)

    assert crew, graph.num_tasks_pending() == 1
    assert crew, graph.num_tasks_remaining() == 1


def test_registered_tasks_are_not_duplicated(crew, graph):
    task = TaskRecipe("", "", crew)
    graph.add_task(task)

    assert graph.num_tasks_pending() == 1
    assert graph.num_tasks_remaining() == 1
