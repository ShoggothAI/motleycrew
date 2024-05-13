from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew import MotleyCrew
from motleycrew.agent.crewai import CrewAIMotleyAgent
from motleycrew.common.utils import configure_logging


def main():
    """Main function of running the example."""
    load_dotenv()

    search_tool = DuckDuckGoSearchRun()

    researcher = CrewAIMotleyAgent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science",
        backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=True,
        delegation=False,
        tools=[search_tool],
    )

    # Create tasks for your agents
    crew = MotleyCrew()
    task = crew.create_simple_task(
        name="produce comprehensive analysis report on AI advancements",
        description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
      Identify key trends, breakthrough technologies, and potential industry impacts.
      Your final answer MUST be a full analysis report""",
        agent=researcher,
    )

    # Get your crew to work!
    result = crew.run(
        verbose=2,  # You can set it to 1 or 2 to different logging levels
    )

    print(task.output)
    return task.output


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
