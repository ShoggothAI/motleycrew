from pathlib import Path
import os
import sys

from dotenv import load_dotenv

from motleycrew.agents.crewai import CrewAIMotleyAgent
from motleycrew.common import configure_logging
from motleycrew.tasks import SimpleTask
from motleycrew.common.exceptions import InvalidOutput

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import StructuredTool

WORKING_DIR = Path(os.path.realpath(".."))

try:
    from motleycrew import MotleyCrew
except ImportError:
    # if we are running this from source
    motleycrew_location = os.path.realpath(WORKING_DIR / "..")
    sys.path.append(motleycrew_location)


def main():
    crew = MotleyCrew()

    search_tool = DuckDuckGoSearchRun()

    def check_output(output: str):
        if "medicine" not in output.lower():
            raise InvalidOutput(
                "Add more information about AI applications in medicine."
            )
        return {"checked_output": output.lower()}

    output_handler = StructuredTool.from_function(
        name="output_handler",
        description="Output handler",
        func=check_output,
    )

    researcher = CrewAIMotleyAgent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science, doing web search if necessary",
        backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You have a knack for dissecting complex data and presenting actionable insights.""",
        delegation=False,
        output_handler=output_handler,
        verbose=True,
        tools=[search_tool],
    )

    # Create tasks for agent

    analysis_report_task = SimpleTask(
        crew=crew,
        name="produce comprehensive analysis report on AI advancements",
        description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts.
    Your final answer MUST be a full analysis report""",
        additional_params={"expected_output": "full analysis report"},
        agent=researcher,
    )

    # Get your crew to work!
    result = crew.run()

    # Get the outputs of the task
    print(analysis_report_task.output)
    return analysis_report_task.output


if __name__ == "__main__":
    configure_logging(verbose=True)
    load_dotenv()
    main()
