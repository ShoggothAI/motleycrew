from pathlib import Path
import os
import sys
import platform

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

import kuzu
from motleycrew.storage import MotleyKuzuGraphStore

from motleycrew import MotleyCrew
from motleycrew.agents.crewai import CrewAIMotleyAgent
from motleycrew.common.utils import configure_logging
from motleycrew.tasks import SimpleTaskRecipe

WORKING_DIR = Path(os.path.realpath("."))

try:
    from motleycrew import MotleyCrew
except ImportError:
    # if we are running this from source
    motleycrew_location = os.path.realpath(WORKING_DIR / "..")
    sys.path.append(motleycrew_location)

if "Dropbox" in WORKING_DIR.parts and platform.system() == "Windows":
    # On Windows, kuzu has file locking issues with Dropbox
    DB_PATH = os.path.realpath(os.path.expanduser("~") + "/Documents/research_db")
else:
    DB_PATH = os.path.realpath(WORKING_DIR / "research_db")


def main():

    db = kuzu.Database(DB_PATH)
    graph_store = MotleyKuzuGraphStore(db)
    crew = MotleyCrew(graph_store=graph_store)

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

    writer = CrewAIMotleyAgent(
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.""",
        verbose=True,
        delegation=True,
    )

    # Create tasks for your agents

    analysis_report_task = SimpleTaskRecipe(
        crew=crew,
        name="produce comprehensive analysis report on AI advancements",
        description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts.
    Your final answer MUST be a full analysis report""",
        agent=researcher,
    )

    literature_summary_task = SimpleTaskRecipe(
        crew=crew,
        name="provide a literature summary of recent papers on AI",
        description="""Conduct a comprehensive literature review of the latest advancements in AI in 2024.
    Identify key papers, researchers, and companies in the space.
    Your final answer MUST be a full literature review with citations""",
        agent=researcher,
    )

    blog_post_task = SimpleTaskRecipe(
        crew=crew,
        name="produce blog post on AI advancements",
        description="""Using the insights provided by a thorough web search, develop an engaging blog
    post that highlights the most significant AI advancements.
    Your post should be informative yet accessible, catering to a tech-savvy audience.
    Make it sound cool, avoid complex words so it doesn't sound like AI.
    Create a blog post of at least 4 paragraphs.""",
        agent=writer,
    )

    [analysis_report_task, literature_summary_task] >> blog_post_task

    # Get your crew to work!
    result = crew.run()

    # Get the outputs of the task
    print(blog_post_task.output)
    return blog_post_task.output


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
