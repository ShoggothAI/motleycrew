from typing import TYPE_CHECKING, List, Set

if TYPE_CHECKING:
    from motleycrew.tasks import TaskRecipe


class TaskGraph:
    def __init__(self):
        self._pending: Set[TaskRecipe] = set()
        self._running: Set[TaskRecipe] = set()
        self._done: Set[TaskRecipe] = set()

    def add_task(self, task: "TaskRecipe"):
        self._pending.add(task)

    def set_task_running(self, task: "TaskRecipe"):
        self._pending.remove(task)
        self._running.add(task)

    def set_task_done(self, task: "TaskRecipe"):
        self._running.remove(task)
        self._done.add(task)
        task.done = True

    def pause_running_task(self, task: "TaskRecipe"):
        self._running.remove(task)
        self._pending.add(task)

    def get_ready_tasks(self) -> List["TaskRecipe"]:
        return [t for t in self._pending if t.is_ready()]

    def check_cyclical_dependencies(self):
        pass

    def num_tasks_remaining(self):
        return len(self._pending) + len(self._running)

    def num_tasks_pending(self):
        return len(self._pending)
