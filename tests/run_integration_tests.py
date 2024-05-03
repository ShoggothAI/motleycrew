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
from examples.single_llama_index import main as single_llama_index_main

CACHE_DIR = "tests/cache"
DATA_DIR = "tests/data"
STRONG_CACHE = True
LOGGING_LEVEL = logging.ERROR

logger = logging.getLogger("integration_test_logger")


class IntegrationTestException(Exception):
    """Integration tests exception"""

    def __init__(self, test_name: str, *args, **kwargs):
        super(IntegrationTestException, self).__init__(*args, **kwargs)
        self.test_name = test_name

    def __str__(self):
        super_str = super(IntegrationTestException, self).__str__()
        return "{} {}: {}".format(self.__class__, self.test_name, super_str)


def single_llama_index_test():
    """Test example single_llama_index"""
    test_name = inspect.stack()[0][3]
    content = single_llama_index_main()
    excepted_content = read_content(test_name)
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


def build_excepted_content_file_path(test_name: str, extension: str = "txt") -> str:
    """Building data file path"""
    return os.path.join(DATA_DIR, "{}.{}".format(test_name, extension))


def write_content(test_name: str, content: str, extension: str = "txt") -> bool:
    """Writing data to file"""
    file_path = build_excepted_content_file_path(test_name, extension)
    with open(file_path, "w") as f_o:
        f_o.write(content)
    return True


def read_content(test_name: str, extension: str = "txt") -> str:
    """Reading data from file"""
    file_path = build_excepted_content_file_path(test_name, extension)
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
