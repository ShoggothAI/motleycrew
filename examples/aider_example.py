from pathlib import Path
import os
import sys

from dotenv import load_dotenv

from langchain_community.tools import ShellTool
from motleycrew.agents.crewai import CrewAIMotleyAgent
from motleycrew.common import configure_logging, AsyncBackend
from motleycrew.tasks import SimpleTask
from motleycache import enable_cache, disable_cache, logger
from motleycrew.tools.aider_tool import AiderTool
import logging

logger.setLevel(logging.INFO)
WORKING_DIR = Path(os.path.realpath("."))

try:
    from motleycrew import MotleyCrew
except ImportError:
    # if we are running this from source
    motleycrew_location = os.path.realpath(WORKING_DIR / "..")
    sys.path.append(motleycrew_location)


# run instruction
# cd ../../
# git clone https://github.com/ShoggothAI/motleycrew-code-generation-example.git


def main():
    crew = MotleyCrew(async_backend=AsyncBackend.THREADING)

    git_dname = r"../../motleycrew-code-generation-example"  # aider coder git dir name
    sys.path.insert(0, os.path.abspath(git_dname))

    check_functions_file = os.path.join(git_dname, "math_functions.py")
    unit_tests_file = os.path.join(git_dname, "test_math_functions.py")
    fnames = [unit_tests_file, check_functions_file]

    aider_tool = AiderTool(fnames=fnames, git_dname=git_dname, auto_commits=False)
    shell_tool = ShellTool()

    researcher = CrewAIMotleyAgent(
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
                        Using py test, you can also add checks for possible exceptions and comments to the tests.""",
        agent=researcher,
    )

    run_unit_tests_task = SimpleTask(
        crew=crew,
        name="Running unit tests",
        description=f"""Run unit tests from the {unit_tests_file} file, using pytest 
        from the current directory and return the execution results""",
        agent=researcher
    )

    create_unit_tests_task >> run_unit_tests_task

    result = crew.run()

    # Get the outputs of the task
    print(run_unit_tests_task.output)
    return run_unit_tests_task.output


if __name__ == "__main__":
    configure_logging(verbose=True)
    load_dotenv()
    main()
