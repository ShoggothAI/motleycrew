"""Thread pool module for running agents"""

from typing import TYPE_CHECKING, Tuple, Any, List
import threading
from queue import Queue
from enum import Enum

from langchain_core.runnables import Runnable

from motleycrew.common import Defaults

if TYPE_CHECKING:
    from motleycrew import Task
    from motleycrew.tasks import TaskUnit


class TaskUnitThreadState(Enum):
    BUSY = "busy"
    WAITING = "waiting"
    EXITED = "exited"


SENTINEL = object()  # sentinel object for closing threads


class TaskUnitThread(threading.Thread):
    def __init__(self, input_queue: Queue, output_queue: Queue, *args, **kwargs):
        """The thread class for running task units

        Args:
            input_queue (Queue): queue of task units to complete
            output_queue (Queue): queue of completed task units
            *args:
            **kwargs:
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._state = TaskUnitThreadState.WAITING

        super(TaskUnitThread, self).__init__(*args, **kwargs)

    @property
    def state(self):
        return self._state

    def run(self) -> None:
        while True:
            run_data = self.input_queue.get()
            self._state = TaskUnitThreadState.BUSY

            if run_data is SENTINEL:
                self._state = TaskUnitThreadState.EXITED
                self.input_queue.task_done()
                break

            agent, task, unit = run_data
            try:
                result = agent.invoke(unit.as_dict())
            except Exception as e:
                self.output_queue.put(e)
            else:
                self.output_queue.put((task, unit, result))
            finally:
                self._state = TaskUnitThreadState.WAITING
                self.input_queue.task_done()


class TaskUnitThreadPool:
    def __init__(self, num_threads: int = Defaults.DEFAULT_NUM_THREADS):
        """The thread pool class for performing task units

        Args:
            num_threads (int): number of threads to create
        """
        self.num_threads = num_threads

        self.input_queue = Queue()
        self.output_queue = Queue()

        self._threads = []
        for i in range(self.num_threads):
            thread = TaskUnitThread(self.input_queue, self.output_queue)
            thread.start()
            self._threads.append(thread)
        self._task_units_in_progress = []

    def add_task_unit(self, agent: Runnable, task: "Task", unit: "TaskUnit"):
        """Adds a task unit to the queue for execution

        Args:
            agent (Runnable):
            task (Task):
            unit (TaskUnit):

        Returns:

        """
        self._task_units_in_progress.append((task, unit))
        self.input_queue.put((agent, task, unit))

    def get_completed_task_units(self) -> List[Tuple["Task", "TaskUnit", Any]]:
        """Returns a list of completed task units with their results

        Returns:
            List[Tuple[Task, TaskUnit, Any]]: list of triplets of task, task unit, and result
        """
        completed_tasks = []
        while not self.output_queue.empty():
            task_result = self.output_queue.get()
            if isinstance(task_result, Exception):
                raise task_result

            task, unit, result = task_result
            completed_tasks.append((task, unit, result))
            self._task_units_in_progress.remove((task, unit))
        return completed_tasks

    def wait_and_close(self):
        """Wait for all task units to complete and close the threads"""
        for t in self._threads:
            if t.is_alive():
                self.input_queue.put(SENTINEL)
        self.input_queue.join()

        for t in self._threads:
            t.join()

    def is_completed(self) -> bool:
        """Returns whether all task units have been completed

        Returns:
            bool:
        """
        return not bool(self._task_units_in_progress)
