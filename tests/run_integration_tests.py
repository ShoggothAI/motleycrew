import os
import sys
from pathlib import Path
import logging
import traceback
import inspect
import difflib as df

from motleycrew.caсhing import (
    enable_cache,
    disable_cache,
    set_cache_location,
    set_strong_cache,
)
from langchain_community.tools import DuckDuckGoSearchRun
from motleycrew import MotleyCrew, Task
from motleycrew.agent.llama_index import ReActLlamaIndexMotleyAgent

CACHE_DIR = "tests/cache"
DATA_DIR = "tests/data"
STRONG_CACHE = True
LOGGING_LEVEL = logging.ERROR

logger = logging.getLogger("integration_test_logger")


class IntegrationTestException(Exception):
    """Integration tests exception"""

    def __init__(self, function_name: str, *args, **kwargs):
        super(IntegrationTestException, self).__init__(*args, **kwargs)
        self.function_name = function_name

    def __str__(self):
        super_str = super(IntegrationTestException, self).__str__()
        return "{} {}: {}".format(self.__class__, self.function_name, super_str)


def single_llama_index_test():
    """Test example single_llama_index"""

    function_name = inspect.stack()[0][3]
    search_tool = DuckDuckGoSearchRun()
    researcher = ReActLlamaIndexMotleyAgent(
        goal="Uncover cutting-edge developments in AI and data science",
        tools=[search_tool],
        verbose=True,
    )
    crew = MotleyCrew()
    # Create tasks for your agents
    task1 = Task(
        crew=crew,
        name="produce comprehensive analysis report on AI advancements",
        description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
      Identify key trends, breakthrough technologies, and potential industry impacts.
      Your final answer MUST be a full analysis report""",
        agent=researcher,
        documents=["paper1.pdf", "paper2.pdf"],
    )

    # Instantiate your crew with a sequential process
    result = crew.run(
        agents=[researcher],
        verbose=2,  # You can set it to 1 or 2 to different logging levels
    )

    content = list(result._done)[0].outputs[0].response
    excepted_content = read_content(function_name)
    comparison_results(content, excepted_content)


def comparison_results(result: str, excepted_result: str) -> list:
    """Comparison of the received and expected results"""
    result_lines = result.splitlines()
    excepted_result_lines = excepted_result.splitlines()
    diff = list(df.unified_diff(result_lines, excepted_result_lines))
    if diff:
        message = "Result content != excepted content.\n{}\n".format(
            "\n".join(diff[3:])
        )
        raise Exception(message)
    return diff


def build_excepted_content_file_path(function_name: str, extension: str = "txt") -> str:
    """Building data file path"""
    return os.path.join(DATA_DIR, "{}.{}".format(function_name, extension))


def write_content(function_name: str, content: str, extension: str = "txt") -> bool:
    """Writing data to file"""
    file_path = build_excepted_content_file_path(function_name, extension)
    with open(file_path, "w") as f_o:
        f_o.write(content)
    return True


def read_content(function_name: str, extension: str = "txt") -> str:
    """Reading data from file"""
    file_path = build_excepted_content_file_path(function_name, extension)
    with open(file_path, "r") as f_o:
        return f_o.read()


def find_tests_functions():
    """Searches for and returns a list of test functions"""
    functions_list = []
    for func_name, func in inspect.getmembers(
        sys.modules[__name__], inspect.isfunction
    ):
        if func_name.endswith("_test"):
            functions_list.append(func)
    return functions_list


if __name__ == "__main__":

    logger.setLevel(LOGGING_LEVEL)

    enable_cache()
    set_cache_location(CACHE_DIR)
    set_strong_cache(STRONG_CACHE)

    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    test_exceptions = []
    test_functions = find_tests_functions()

    for f in test_functions:
        try:
            logger.info("Start function: {}".format(f.__name__))
            f()
        except Exception as e:
            msg = "{}\n{}".format(str(e), traceback.format_exc())
            test_exceptions.append(IntegrationTestException(f.__name__, msg))
            continue

    for i, t_e in enumerate(test_exceptions):
        if i == len(test_exceptions) - 1:
            disable_cache()
            raise t_e
        logger.error(str(t_e))
    disable_cache()
