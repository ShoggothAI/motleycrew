import pytest

from motleycrew.crew.crew_threads import TaskUnitThreadPool, TaskUnitThreadState
from motleycrew.common import Defaults
from tests.test_crew import CrewFixtures


class TestInvokeThreadPool(CrewFixtures):

    @pytest.fixture
    def thread_pool(self):
        obj = TaskUnitThreadPool()
        yield obj
        obj.wait_and_close()

    @pytest.mark.parametrize("tasks", [4], indirect=True)
    @pytest.fixture
    def thread_pool_with_tasks(self, tasks, thread_pool, agent):
        for task in tasks:
            unit = task.get_next_unit()
            thread_pool.add_task_unit(agent, task, unit)

        return thread_pool

    def test_init_thread_pool(self, thread_pool):
        assert len(thread_pool._threads) == Defaults.DEFAULT_NUM_THREADS
        assert all([t.is_alive() for t in thread_pool._threads])
        assert thread_pool.input_queue.empty()
        assert thread_pool.output_queue.empty()
        assert thread_pool.is_completed

    @pytest.mark.parametrize("tasks", [4], indirect=True)
    def test_put(self, thread_pool, agent, tasks):
        for task in tasks:
            unit = task.get_next_unit()
            thread_pool.add_task_unit(agent, task, unit)

        assert not thread_pool.is_completed
        assert len(thread_pool._task_units_in_progress) == 4

    @pytest.mark.parametrize("tasks", [4], indirect=True)
    def test_get_completed_tasks(self, thread_pool, agent, tasks):
        for task in tasks:
            unit = task.get_next_unit()
            thread_pool.add_task_unit(agent, task, unit)

        thread_pool.wait_and_close()
        completed_tasks = thread_pool.get_completed_task_units()

        assert len(completed_tasks) == 4
        assert len(thread_pool._task_units_in_progress) == 0
        assert thread_pool.is_completed
        assert all([t.state == TaskUnitThreadState.EXITED for t in thread_pool._threads])

    @pytest.mark.parametrize("tasks", [1], indirect=True)
    def test_get_completed_task_exception(self, thread_pool, agent, tasks):
        for task in tasks:
            thread_pool.add_task_unit(agent, task, None)
        thread_pool.wait_and_close()

        with pytest.raises(AttributeError):
            thread_pool.get_completed_task_units()

        assert not thread_pool.is_completed

    def test_close(self, thread_pool):
        thread_pool.wait_and_close()
        assert all([not t.is_alive() for t in thread_pool._threads])
        assert all([t.state == TaskUnitThreadState.EXITED for t in thread_pool._threads])
