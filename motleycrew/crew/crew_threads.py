"""Thread pool module for running agents"""

from typing import Tuple, Any, List
import threading
from queue import Queue
from enum import Enum


class InvokeThreadState(Enum):
    PROCESS = "process"
    WAITING = "waiting"
    STOP = "stop"


class InvokeThread(threading.Thread):

    def __init__(self, input_queue: Queue, output_queue: Queue, *args, **kwargs):
        """The thread class for running tasks

        Args:
            input_queue (Queue): queue of tasks to complete
            output_queue (Queue): queue of completed tasks
            *args:
            **kwargs:
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.__state = InvokeThreadState.WAITING
        self.__is_close = False
        super(InvokeThread, self).__init__(*args, **kwargs)

    @property
    def state(self):
        return self.__state

    def stop(self):
        """Plans to stop the execution cycle when the next task is received"""
        self.input_queue.put(None)
        self.__is_close = True

    def run(self) -> None:
        while True:
            running_data = self.input_queue.get()

            self.__state = InvokeThreadState.PROCESS

            if running_data is None or self.__is_close:
                self.__state = InvokeThreadState.STOP
                break

            agent, task, unit = running_data
            result = agent.invoke(unit.as_dict())
            self.output_queue.put((task, unit, result))
            self.__state = InvokeThreadState.WAITING


class InvokeThreadPool:

    def __init__(self, num_threads: int = None):
        """The thread storage class for performing tasks

        Args:
            num_threads (int): number of threads being created default 4
        """
        self.num_threads = num_threads or 4

        self.lock = threading.Lock()
        self.input_queue = Queue()
        self.output_queue = Queue()

        self.__threads = []
        for i in range(self.num_threads):
            thread = InvokeThread(self.input_queue, self.output_queue)
            thread.start()
            self.__threads.append(thread)

    def put(self, agent: "Runnable", task: "Task", unit: "TaskUnit"):
        """Adds a task to the queue for execution

        Args:
            agent (Runnable):
            task (Task):
            unit (TaskUnit):

        Returns:

        """
        self.input_queue.put((agent, task, unit))

    def get_completed_tasks(self) -> List[Tuple["Task", "TaskUnit", Any]]:
        """Returns a list of completed tasks

        Returns:
            List[Tuple[Task, TaskUnit, Any]]: list of completed tasks  or empty list
        """
        completed_tasks = []
        while not self.output_queue.empty():
            completed_tasks.append(self.output_queue.get())
        return completed_tasks

    def close(self):
        """Closes running threads"""
        for t in self.__threads:
            t.stop()

    def is_completed(self) -> bool:
        """Returns the result of checking the completion and returning all tasks

        Returns:
            bool:
        """
        self.lock.acquire()

        in_process = any([t.state == InvokeThreadState.PROCESS for t in self.__threads])
        empty_queue = bool(self.input_queue.empty() and self.output_queue.empty())
        is_completed = bool(not in_process and empty_queue)

        self.lock.release()
        return is_completed
