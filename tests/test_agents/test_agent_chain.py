import pytest

from motleycrew.agents.parent import MotleyAgentParent


class AgentMock:
    def invoke(self, input_dict: dict, *args, **kwargs):
        return input_dict


class MotleyAgentMock(MotleyAgentParent):

    def invoke(self, *args, **kwargs):
        self.materialize()
        return self.agent.invoke(*args, **kwargs)


def agent_factory(*args, **kwargs):
    return AgentMock()


@pytest.fixture
def motley_agents():
    agent1 = MotleyAgentMock("agent1 description", agent_factory=agent_factory)
    agent2 = MotleyAgentMock("agent2 description", agent_factory=agent_factory)
    return [agent1, agent2]


def test_agent_chain(motley_agents):
    agent1, agent2 = motley_agents
    agent_chain = agent1 | agent2
    assert hasattr(agent_chain, "invoke")
    prompt = {"prompt": "test_prompt"}
    assert agent_chain.invoke(prompt) == prompt
