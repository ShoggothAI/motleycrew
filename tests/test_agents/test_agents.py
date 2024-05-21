import os
import pytest

from langchain_community.tools import DuckDuckGoSearchRun
from motleycrew.agents.crewai.crewai_agent import CrewAIMotleyAgent
from motleycrew.agents.langchain.react import ReactMotleyAgent
from motleycrew.agents.llama_index.llama_index_react import ReActLlamaIndexMotleyAgent
from motleycrew.common.exceptions import AgentNotMaterialized, CannotModifyMaterializedAgent
from motleycrew.tools.python_repl import create_repl_tool
from motleycrew.tools.tool import MotleyTool

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"

test_agents_names = ("crewai", "langchain", "llama_index")


class TestAgents:

    @pytest.fixture(scope="class")
    def crewai_agent(self):
        agent = CrewAIMotleyAgent(
            role="Senior Research Analyst",
            goal="Uncover cutting-edge developments in AI and data science",
            backstory="""You work at a leading tech think tank.
           Your expertise lies in identifying emerging trends.
           You have a knack for dissecting complex data and presenting actionable insights.""",
            verbose=True,
            delegation=False,
            tools=[DuckDuckGoSearchRun()],
        )
        return agent

    @pytest.fixture(scope="class")
    def langchain_agent(self):
        agent = ReactMotleyAgent(
            name="AI writer agent",
            description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
                  Identify key trends, breakthrough technologies, and potential industry impacts.
                  Your final answer MUST be a full analysis report""",
            tools=[DuckDuckGoSearchRun()],
            verbose=True,
        )
        return agent

    @pytest.fixture(scope="class")
    def llama_index_agent(self):
        agent = ReActLlamaIndexMotleyAgent(
            description="Uncover cutting-edge developments in AI and data science",
            tools=[DuckDuckGoSearchRun()],
            verbose=True,
        )
        return agent

    @pytest.fixture(scope="class")
    def agent(self, request, crewai_agent, langchain_agent, llama_index_agent):
        agents = {
            "crewai": crewai_agent,
            "langchain": langchain_agent,
            "llama_index": llama_index_agent,
        }
        return agents.get(request.param)

    @pytest.mark.parametrize("agent", test_agents_names, indirect=True)
    def test_add_tools(self, agent):
        assert len(agent.tools) == 1
        tools = [DuckDuckGoSearchRun()]
        agent.add_tools(tools)
        assert len(agent.tools) == 1

    @pytest.mark.parametrize("agent", test_agents_names, indirect=True)
    def test_materialized(self, agent):
        with pytest.raises(AgentNotMaterialized):
            agent.agent

        assert not agent.is_materialized
        agent.materialize()
        assert agent.is_materialized

        with pytest.raises(CannotModifyMaterializedAgent):
            tool = create_repl_tool()
            agent.add_tools([tool])

    @pytest.mark.parametrize("agent", test_agents_names, indirect=True)
    def test_as_tool(self, agent):
        tool = agent.as_tool()
        assert isinstance(tool, MotleyTool)
