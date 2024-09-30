import os

import pytest
from langchain_core.prompts.chat import ChatPromptTemplate

from motleycrew.agents.crewai.crewai_agent import CrewAIMotleyAgent
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.agents.llama_index.llama_index_react import ReActLlamaIndexMotleyAgent
from motleycrew.common.exceptions import (
    AgentNotMaterialized,
    CannotModifyMaterializedAgent,
)
from motleycrew.tools.code.python_repl import create_repl_tool
from tests.test_agents import MockTool

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"

test_agents_names = ("crewai", "langchain", "llama_index")


class TestAgents:

    @pytest.fixture(scope="class")
    def crewai_agent(self):
        agent = CrewAIMotleyAgent(
            role="Senior Research Analyst",
            goal="Uncover cutting-edge developments in AI and data science",
            prompt_prefix="""You work at a leading tech think tank.
           Your expertise lies in identifying emerging trends.
           You have a knack for dissecting complex data and presenting actionable insights.""",
            backstory="",
            verbose=True,
            delegation=False,
            tools=[MockTool()],
        )
        return agent

    @pytest.fixture(scope="class")
    def langchain_agent(self):
        agent = ReActToolCallingMotleyAgent(
            name="AI writer agent",
            prompt_prefix="Generate AI-generated content",
            description="AI-generated content",
            tools=[MockTool()],
            verbose=True,
        )
        return agent

    @pytest.fixture(scope="class")
    def llama_index_agent(self):
        agent = ReActLlamaIndexMotleyAgent(
            prompt_prefix="Uncover cutting-edge developments in AI and data science",
            description="AI researcher",
            tools=[MockTool()],
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
        tools = [MockTool()]
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
    def test_compose_prompt(self, agent):
        task_prompt = ChatPromptTemplate.from_template("What are the latest {topic} trends?")
        task_dict = {"topic": "AI"}
        prompt = agent.compose_prompt(input_dict=task_dict, prompt=task_prompt)

        assert str(agent.prompt_prefix) in prompt
        assert "What are the latest AI trends?" in prompt
