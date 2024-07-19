import os
import platform
import sys
from pathlib import Path

import kuzu
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew.agents.crewai import CrewAIMotleyAgent
from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
from motleycrew.agents.llama_index import ReActLlamaIndexMotleyAgent
from motleycrew.common import configure_logging
from motleycrew.storage import MotleyKuzuGraphStore
from motleycrew.tasks import SimpleTask
from motleycrew.tools.image.dall_e import DallEImageGeneratorTool

WORKING_DIR = Path(os.path.realpath("."))

try:
    from motleycrew import MotleyCrew
except ImportError:
    # if we are running this from source
    motleycrew_location = os.path.realpath(WORKING_DIR / "..")
    sys.path.append(motleycrew_location)

if __name__ == "__main__":
    if "Dropbox" in WORKING_DIR.parts and platform.system() == "Windows":
        # On Windows, kuzu has file locking issues with Dropbox
        DB_PATH = os.path.realpath(os.path.expanduser("~") + "/Documents/research_db")
    else:
        DB_PATH = os.path.realpath(WORKING_DIR / "research_db")

else:
    DB_PATH = os.path.realpath(WORKING_DIR / "tests/research_db")


def main():

    db = kuzu.Database(DB_PATH)
    graph_store = MotleyKuzuGraphStore(db)
    crew = MotleyCrew(graph_store=graph_store)

    search_tool = DuckDuckGoSearchRun()

    researcher = CrewAIMotleyAgent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science, doing web search if necessary",
        backstory="""You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=True,
        tools=[search_tool],
    )

    # You can give agents as tools to other agents
    writer = ReActToolCallingMotleyAgent(
        name="AI writer agent",
        prompt_prefix="You are an experienced writer with a passion for technology.",
        description="Experienced writer with a passion for technology.",
        tools=[researcher],
        verbose=True,
    )

    # Illustrator
    illustrator = ReActLlamaIndexMotleyAgent(
        name="Illustrator",
        prompt_prefix="Create beautiful and insightful illustrations for a blog post",
        tools=[DallEImageGeneratorTool(os.path.realpath("./images"))],
    )

    blog_post_task = SimpleTask(
        crew=crew,
        name="produce blog post on AI advancements",
        description="""Using the insights provided by a thorough web search, develop an engaging blog
    post that highlights the most significant AI advancements.
    Your post should be informative yet accessible, catering to a tech-savvy audience.
    Make it sound cool, avoid complex words so it doesn't sound like AI.
    Create a blog post of at least 4 paragraphs, in markdown format.""",
        agent=writer,
    )

    illustration_task = SimpleTask(
        crew=crew,
        name="create an illustration for the blog post",
        description="""Create beautiful and insightful illustrations to accompany the blog post on AI advancements.
        The blog post will be provided to you in markdown format.
        Make sure to use the illustration tool provided to you, once per illustration, and embed the URL provided by
        the tool into the blog post.""",
        agent=illustrator,
    )

    # Make sure the illustration task runs only once the blog post task is complete, and gets its input
    blog_post_task >> illustration_task

    # Get your crew to work!
    result = crew.run()

    # Get the outputs of the task
    print(blog_post_task.output)
    print(illustration_task.output)
    return illustration_task.output


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
    print("yay!")
