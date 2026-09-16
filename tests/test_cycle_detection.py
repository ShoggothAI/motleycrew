"""Test cycle detection in add_dependency."""

import pytest

from motleycrew.common.exceptions import TaskDependencyCycleError
from motleycrew.crew.crew import MotleyCrew
from motleycrew.tasks.task import Task, TaskNode
from motleycrew.tasks.task_unit import TaskUnit


class MockTaskUnit(TaskUnit):
    """Minimal task unit for testing."""
    __label__ = "MockTaskUnit"
    
    def __init__(self, name: str = "test"):
        super().__init__(name=name, prompt="test prompt")


class MockTask(Task):
    """Minimal task for testing cycle detection."""
    NODE_CLASS = TaskNode
    
    def __init__(self, name: str, crew=None):
        super().__init__(name=name, task_unit_class=MockTaskUnit, crew=crew)
        self._worker = None
    
    def get_next_unit(self):
        return None
    
    def get_worker(self, tools=None):
        return self._worker


def test_add_dependency_allows_valid_chain():
    """A -> B -> C should be allowed."""
    crew = MotleyCrew()
    task_a = MockTask("A", crew=crew)
    task_b = MockTask("B", crew=crew)
    task_c = MockTask("C", crew=crew)
    
    crew.add_dependency(task_a, task_b)
    crew.add_dependency(task_b, task_c)
    
    # Verify the chain
    assert task_b in task_c.get_upstream_tasks()
    assert task_a in task_b.get_upstream_tasks()


def test_add_dependency_rejects_direct_cycle():
    """A -> B -> A should be rejected."""
    crew = MotleyCrew()
    task_a = MockTask("A", crew=crew)
    task_b = MockTask("B", crew=crew)
    
    crew.add_dependency(task_a, task_b)
    
    with pytest.raises(TaskDependencyCycleError):
        crew.add_dependency(task_b, task_a)


def test_add_dependency_rejects_transitive_cycle():
    """A -> B -> C -> A should be rejected."""
    crew = MotleyCrew()
    task_a = MockTask("A", crew=crew)
    task_b = MockTask("B", crew=crew)
    task_c = MockTask("C", crew=crew)
    
    crew.add_dependency(task_a, task_b)
    crew.add_dependency(task_b, task_c)
    
    with pytest.raises(TaskDependencyCycleError):
        crew.add_dependency(task_c, task_a)


def test_add_dependency_rejects_self_loop():
    """A -> A should be rejected (existing behavior)."""
    crew = MotleyCrew()
    task_a = MockTask("A", crew=crew)
    
    with pytest.raises(TaskDependencyCycleError):
        task_a.set_upstream(task_a)


def test_add_dependency_allows_diamond():
    """A -> B, A -> C, B -> D, C -> D should be allowed (diamond, not cycle)."""
    crew = MotleyCrew()
    task_a = MockTask("A", crew=crew)
    task_b = MockTask("B", crew=crew)
    task_c = MockTask("C", crew=crew)
    task_d = MockTask("D", crew=crew)
    
    crew.add_dependency(task_a, task_b)
    crew.add_dependency(task_a, task_c)
    crew.add_dependency(task_b, task_d)
    crew.add_dependency(task_c, task_d)
    
    # D should have B and C as upstreams
    upstreams = task_d.get_upstream_tasks()
    assert task_b in upstreams
    assert task_c in upstreams
