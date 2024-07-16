"""Thread pool module for running agents."""

import threading
from enum import Enum
from queue import Queue
from typing import TYPE_CHECKING, Tuple, Any, List

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
    """The thread class for running agents on task units."""

    def __init__(self, input_queue: Queue, output_queue: Queue, *args, **kwargs):
        """Initialize the thread.

        Args:
            input_queue: Queue of task units to complete.
            output_queue: Queue of completed task units.
            *args: threading.Thread arguments.
            **kwargs: threading.Thread keyword arguments.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._state = TaskUnitThreadState.WAITING

        super(TaskUnitThread, self).__init__(*args, **kwargs)

    @property
    def state(self):
        """State of the thread."""
        return self._state

    def run(self) -> None:
        """Main loop of the thread.

        Gets a task unit from the input queue, runs it, and puts the result in the output queue.
        Exits when the sentinel object is retrieved from the input queue.
        """
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
    """The thread pool class for running agents on task units."""

    def __init__(self, num_threads: int = Defaults.DEFAULT_NUM_THREADS):
        """Initialize the thread pool.

        Args:
            num_threads: Number of threads to create.
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

    def add_task_unit(self, agent: Runnable, task: "Task", unit: "TaskUnit") -> None:
        """Adds a task unit to the queue for execution.

        Args:
            agent: Agent to run the task unit.
            task: Task to which the unit belongs.
            unit: Task unit to run.
        """
        self._task_units_in_progress.append((task, unit))
        self.input_queue.put((agent, task, unit))

    def get_completed_task_units(self) -> List[Tuple["Task", "TaskUnit", Any]]:
        """Returns a list of completed task units with their results.

        Returns:
            List of triplets of (task, task unit, result).
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
        """Wait for all task units to complete and close the threads."""
        for t in self._threads:
            if t.is_alive():
                self.input_queue.put(SENTINEL)
        self.input_queue.join()

        for t in self._threads:
            t.join()

    @property
    def is_completed(self) -> bool:
        """Whether all task units have been completed."""
        return not bool(self._task_units_in_progress)
