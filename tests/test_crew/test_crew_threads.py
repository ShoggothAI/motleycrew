import pytest
import time

from motleycrew.crew.crew_threads import InvokeThreadPool, InvokeThreadState
from motleycrew.common import Defaults
from tests.test_crew import CrewFixtures


class TestInvokeThreadPool(CrewFixtures):

    @pytest.fixture(scope="class")
    def thread_pool(self):
        obj = InvokeThreadPool()
        return obj

    def test_init_thread_pool(self, thread_pool):

        assert len(thread_pool._threads) == Defaults.DEFAULT_NUM_THREADS
        assert all([t.is_alive() for t in thread_pool._threads])
        assert thread_pool.input_queue.empty()
        assert thread_pool.output_queue.empty()
        assert thread_pool.is_completed()

    @pytest.mark.parametrize("tasks", [4], indirect=True)
    def test_put(self, thread_pool, agent, tasks):
        for task in tasks:
            unit = task.get_next_unit()
            thread_pool.put(agent, task, unit)

        assert not thread_pool.is_completed()
        assert len(thread_pool._in_process_tasks) == 4

    def test_get_completed_tasks(self, thread_pool):
        time.sleep(3)
        completed_tasks = thread_pool.get_completed_tasks()
        assert len(completed_tasks) == 4
        assert len(thread_pool._in_process_tasks) == 0
        assert thread_pool.is_completed()
        assert all([t.state == InvokeThreadState.WAITING for t in thread_pool._threads])

    @pytest.mark.parametrize("tasks", [1], indirect=True)
    def test_get_completed_task_exception(self, thread_pool, agent, tasks):
        for task in tasks:
            thread_pool.put(agent, task, None)
        time.sleep(1)

        with pytest.raises(AttributeError):
            thread_pool.get_completed_tasks()

        assert not thread_pool.is_completed()

    def test_close(self, thread_pool):
        thread_pool.close()
        time.sleep(3)
        assert all([not t.is_alive() for t in thread_pool._threads])
        assert all([t.state == InvokeThreadState.STOP for t in thread_pool._threads])

    def test_is_completed(self, thread_pool):
        assert len(thread_pool._in_process_tasks) == 1
        assert not thread_pool.is_completed()
