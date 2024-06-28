import uuid
from collections import deque
import pytest

from langchain_community.tools import DuckDuckGoSearchRun

try:
    from llama_index.core.agent.types import Task, TaskStep, TaskStepOutput
    from llama_index.core.chat_engine.types import AgentChatResponse
    from llama_index.core.agent.runner.base import TaskState
except ImportError:
    Task = None
    TaskStep = None
    TaskStepOutput = None
    AgentChatResponse = None
    TaskState = None

from motleycrew.agents.llama_index import ReActLlamaIndexMotleyAgent
from motleycrew.agents import MotleyOutputHandler
from motleycrew.common.exceptions import InvalidOutput, ModuleNotInstalled


class ReportOutputHandler(MotleyOutputHandler):
    def handle_output(self, output: str):
        if "medical" not in output.lower():
            raise InvalidOutput("Add more information about AI applications in medicine.")

        return {"checked_output": output}


def fake_run_step(*args, **kwargs):
    task_step_output = kwargs.get("task_step_output")
    return task_step_output


@pytest.fixture
def agent():
    try:
        search_tool = DuckDuckGoSearchRun()
        agent = ReActLlamaIndexMotleyAgent(
            description="Your goal is to uncover cutting-edge developments in AI and data science",
            tools=[search_tool],
            output_handler=ReportOutputHandler(),
            verbose=True,
        )
        agent.materialize()
        agent._agent._run_step = fake_run_step
        agent._agent._run_step = agent.run_step_decorator()(agent._agent._run_step)

    except ModuleNotInstalled:
        return
    return agent


def test_run_step(agent):
    if agent is None:
        return

    task = Task(input="User input", memory=agent._agent.memory)
    task_step = TaskStep(task_id=task.task_id, step_id=str(uuid.uuid4()), input="Test input")

    task_state = TaskState(
        task=task,
        step_queue=deque([task_step]),
    )
    agent._agent.state.task_dict[task.task_id] = task_state

    task_step_output = TaskStepOutput(
        is_last=False,
        task_step=task_step,
        output=AgentChatResponse(response="Test response"),
        next_steps=[],
    )

    # test not last output
    cur_step_output = agent._agent._run_step("", task_step_output=task_step_output)
    assert task_step_output == cur_step_output
    assert not cur_step_output.next_steps

    # test is last output
    task_step_output.is_last = True
    cur_step_output = agent._agent._run_step("", task_step_output=task_step_output)

    assert task_step_output == cur_step_output
    assert not cur_step_output.is_last
    assert cur_step_output.next_steps

    step_queue = agent._agent.state.get_step_queue(task.task_id)
    _task_step = step_queue.pop()

    assert _task_step.task_id == task.task_id
    assert _task_step.input == "You must call the {} tool to return the output.".format(
        agent.output_handler.name
    )
