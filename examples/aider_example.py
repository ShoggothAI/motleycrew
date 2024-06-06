from pathlib import Path
import os
import sys

from dotenv import load_dotenv
import logging


from langchain_community.tools import ShellTool
from motleycrew.agents.crewai import CrewAIMotleyAgent
from motleycrew.common import configure_logging, AsyncBackend
from motleycrew.tasks import SimpleTask
from motleycache import logger
from motleycrew.tools.aider_tool import AiderTool

from aider.io import InputOutput

logger.setLevel(logging.INFO)
WORKING_DIR = Path(os.path.realpath("."))

try:
    from motleycrew import MotleyCrew
except ImportError:
    # if we are running this from source
    motleycrew_location = os.path.realpath(WORKING_DIR / "..")
    sys.path.append(motleycrew_location)


# run instruction
# to run the example, you need to clone the repository for the example at the same level as your project
# cd ../../
# git clone https://github.com/ShoggothAI/motleycrew-code-generation-example.git


def main():
    crew = MotleyCrew(async_backend=AsyncBackend.THREADING)

    git_dname = r"../../motleycrew-code-generation-example"  # aider coder git dir name
    io = InputOutput(yes=True)

    aider_tool = AiderTool(io=io, git_dname=git_dname, auto_commits=False)
    shell_tool = ShellTool()

    developer = CrewAIMotleyAgent(
        role="Software Engineer",
        goal="Writing unit tests",
        backstory="""You are a leading software engineer working in the field of computer technology""",
        delegation=False,
        verbose=True,
        tools=[aider_tool, shell_tool],
    )

    # Create tasks for your agent
    create_unit_tests_task = SimpleTask(
        crew=crew,
        name="Adding a unit test",
        description=f"""Generate unit tests for the module math_functions.py
                        Using py test, you can also add checks for possible exceptions and comments to the tests
                        and using parameterization for test functions .
                        Go to the directory {git_dname} and run created unit tests for math_functions.
                        If the tests were executed successfully, return the result of execution, 
                        if not, rewrite the tests and restart them.
                        If the command to run unit tests fails, try to run it one more time, 
                        despite the fact that this command has already been executed. 
                        """,
        agent=developer,
    )

    result = crew.run()

    # Get the outputs of the task
    print(create_unit_tests_task.output)
    return create_unit_tests_task.output


if __name__ == "__main__":
    configure_logging(verbose=True)
    load_dotenv()
    main()
