import uuid
from collections import deque

import pytest
from langchain_core.tools import StructuredTool

try:
    from llama_index.core.agent.runner.base import TaskState
    from llama_index.core.agent.types import Task, TaskStep, TaskStepOutput
    from llama_index.core.chat_engine.types import AgentChatResponse
except ImportError:
    Task = None
    TaskStep = None
    TaskStepOutput = None
    AgentChatResponse = None
    TaskState = None

from motleycrew.agents.llama_index import ReActLlamaIndexMotleyAgent
from motleycrew.common import AuxPrompts
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.tools import MotleyTool
from tests.test_agents import MockTool

invalid_output = "Add more information about AI applications in medicine."


class ReportOutputHandler(MotleyTool):
    def __init__(self):
        super().__init__(
            name="output_handler",
            description="Output handler",
            return_direct=True,
        )

    def run(self, output: str):
        if "medical" not in output.lower():
            raise InvalidOutput(invalid_output)

        return {"checked_output": output}


def fake_run_step(*args, **kwargs):
    task_step_output = kwargs.get("task_step_output")
    output_handler = kwargs.get("output_handler")
    output_handler_input = kwargs.get("output_handler_input")
    if output_handler:
        output_handler_result = output_handler.invoke(output_handler_input)
        task_step_output.output = AgentChatResponse(response=output_handler_result)

    return task_step_output


@pytest.fixture
def agent():

    agent = ReActLlamaIndexMotleyAgent(
        description="Your goal is to uncover cutting-edge developments in AI and data science",
        tools=[MockTool(), ReportOutputHandler()],
        force_output_handler=True,
        verbose=True,
    )
    agent.materialize()
    agent._agent._run_step = fake_run_step
    agent._agent._run_step = agent._run_step_decorator()(agent._agent._run_step)

    return agent


@pytest.fixture
def task_data(agent):
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
    return task, task_step_output


def find_output_handler(agent: ReActLlamaIndexMotleyAgent) -> StructuredTool:
    agent_worker = agent.agent.agent_worker
    output_handler = None
    for tool in agent_worker._get_tools(""):
        if tool.metadata.name == "output_handler":
            output_handler = tool.to_langchain_tool()
            break
    return output_handler


def test_run_step(agent, task_data):
    if agent is None:
        return

    task, task_step_output = task_data

    # test not last output
    cur_step_output = agent._agent._run_step("", task_step_output=task_step_output)
    assert task_step_output == cur_step_output
    assert not cur_step_output.next_steps

    # test is last output
    task_step_output.is_last = True
    cur_step_output = agent._agent._run_step("", task_step_output=task_step_output)

    assert task_step_output == cur_step_output
    assert not cur_step_output.is_last

    step_queue = agent._agent.state.get_step_queue(task.task_id)
    _task_step = step_queue.pop()

    assert _task_step.task_id == task.task_id
    assert _task_step.input == AuxPrompts.get_direct_output_error_message(
        agent.get_output_handlers()
    )

    # test direct output
    output_handler = ReportOutputHandler()

    # test wrong output
    output_handler_input = "Latest advancements in AI in 2024."
    cur_step_output = agent._agent._run_step(
        "",
        task_step_output=task_step_output,
        output_handler=output_handler,
        output_handler_input=output_handler_input,
    )
    assert cur_step_output.output.response == "InvalidOutput: {}".format(invalid_output)

    # test correct output
    output_handler_input = "Latest advancements in medical AI in 2024."
    cur_step_output = agent._agent._run_step(
        "",
        task_step_output=task_step_output,
        output_handler=output_handler,
        output_handler_input=output_handler_input,
    )
    assert cur_step_output.is_last
    assert cur_step_output.output.response == "{{'checked_output': '{}'}}".format(
        output_handler_input
    )
    assert hasattr(agent, "direct_output")
    assert agent.direct_output.output == {"checked_output": output_handler_input}
