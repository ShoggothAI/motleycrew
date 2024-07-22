import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.tools import ShellTool
from motleycache import logger

from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.common import configure_logging
from motleycrew.tasks import SimpleTask
from motleycrew.tools.code.aider_tool import AiderTool

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
    crew = MotleyCrew()

    git_repo_path = r"../../motleycrew-code-generation-example"  # cloned repository path
    tests_file = os.path.join(git_repo_path, "test_math_functions.py")
    target_files = [tests_file]

    aider_tool = AiderTool(fnames=target_files, git_dname=git_repo_path, auto_commits=False)
    shell_tool = ShellTool()

    developer = ReActToolCallingMotleyAgent(
        prompt_prefix="You are a lead software engineer working in a big tech company.",
        verbose=True,
        tools=[aider_tool, shell_tool],
    )

    create_unit_tests_task = SimpleTask(
        crew=crew,
        name="Adding a unit test",
        description=f"Generate unit tests for the module math_functions.py using pytest. "
        f"You should also add test cases for possible exceptions "
        f"and write comments to the tests. You should also use test parameterization. "
        f"After go to the directory {git_repo_path} and run created unit tests. "
        f"If the tests were executed successfully, return the result of execution, "
        f"if not, rewrite the tests and rerun them until they are working.",
        additional_params={"expected_output": "result of tests execution"},
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
