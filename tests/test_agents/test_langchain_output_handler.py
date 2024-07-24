import pytest
from langchain_core.agents import AgentFinish, AgentAction

from motleycrew.agents import MotleyOutputHandler
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.agents.parent import DirectOutput
from motleycrew.common.exceptions import InvalidOutput, OutputHandlerMaxIterationsExceeded
from tests.test_agents import MockTool

invalid_output = "Add more information about AI applications in medicine."


class ReportOutputHandler(MotleyOutputHandler):
    def handle_output(self, output: str):
        if "medical" not in output.lower():
            raise InvalidOutput(invalid_output)

        return {"checked_output": output}


def fake_agent_plan(intermediate_steps, step, **kwargs):
    return step


def fake_agent_take_next_step(
    name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager
):

    output_handler = name_to_tool_map.get("output_handler")
    result = output_handler._run(inputs)

    if isinstance(result, DirectOutput):
        raise result

    return result


@pytest.fixture
def agent():
    agent = ReActToolCallingMotleyAgent(
        tools=[MockTool()],
        verbose=True,
        chat_history=True,
        output_handler=ReportOutputHandler(max_iterations=5),
    )
    agent.materialize()
    object.__setattr__(agent._agent, "plan", fake_agent_plan)
    object.__setattr__(agent.agent, "plan", agent.agent_plan_decorator(agent.agent.plan))

    object.__setattr__(agent._agent, "_take_next_step", fake_agent_take_next_step)
    object.__setattr__(
        agent._agent,
        "_take_next_step",
        agent.take_next_step_decorator(agent.agent._take_next_step),
    )
    return agent


@pytest.fixture
def run_kwargs(agent):
    agent_executor = agent.agent.bound.bound.last.default.bound

    run_kwargs = {
        "name_to_tool_map": {tool.name: tool for tool in agent_executor.tools},
        "color_mapping": {},
        "inputs": {},
        "intermediate_steps": [],
    }
    return run_kwargs


def test_agent_plan(agent):
    agent_executor = agent.agent
    agent_action = AgentAction("tool", "tool_input", "tool_log")
    step = agent_executor.plan([], agent_action)
    assert agent_action == step

    return_values = {"output": "test_output"}
    agent_finish = AgentFinish(return_values=return_values, log="test_output")

    step = agent_executor.plan([], agent_finish)
    assert isinstance(step, AgentAction)
    assert step.tool == agent._agent_finish_blocker_tool.name
    assert step.tool_input == "test_output"


def test_agent_take_next_step(agent, run_kwargs):

    # test wrong output
    input_data = "Latest advancements in AI in 2024."
    run_kwargs["inputs"] = input_data
    step_result = agent.agent._take_next_step(**run_kwargs)
    assert step_result == f"{InvalidOutput.__name__}: {invalid_output}"

    # test correct output
    input_data = "Latest advancements in medical AI in 2024."
    run_kwargs["inputs"] = input_data
    step_result = agent.agent._take_next_step(**run_kwargs)
    assert isinstance(step_result, AgentFinish)
    assert isinstance(step_result.return_values, dict)
    output_result = step_result.return_values.get("output")
    assert output_result == {"checked_output": input_data}


def test_output_handler_max_iteration(agent, run_kwargs):
    input_data = "Latest advancements in AI in 2024."
    run_kwargs["inputs"] = input_data

    with pytest.raises(OutputHandlerMaxIterationsExceeded):
        for iteration in range(agent.output_handler.max_iterations + 1):
            agent.agent._take_next_step(**run_kwargs)

    assert iteration == agent.output_handler.max_iterations
