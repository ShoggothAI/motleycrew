import argparse
import os
import sys
import traceback
from copy import copy
from functools import partial
from pathlib import Path
from typing import Optional

import nbformat
from dotenv import load_dotenv
from motleycache import set_cache_location
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4.nbbase import new_code_cell

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)  # ensure the examples are importable

from motleycrew.common import logger, configure_logging
from motleycrew.common.exceptions import (
    IntegrationTestException,
    IpynbIntegrationTestResultNotFound,
)

INTEGRATION_TESTS = {}

IPYNB_INTEGRATION_TESTS = {
    # "blog_with_images_ipynb": "examples/Blog with Images.ipynb",
    "multi_step_research_agent_ipynb": "examples/Multi-step research agent.ipynb",
    "math_via_python_code_with_a_single_agent_ipynb": "examples/Math via python code with a single agent.ipynb",
    "validating_agent_output_ipynb": "examples/Validating agent output.ipynb",
    "advanced_output_handling_ipynb": "examples/Advanced output handling.ipynb",
    "using_autogen_with_motleycrew_ipynb": "examples/Using AutoGen with motleycrew.ipynb",
}

INTEGRATION_TESTS_TO_SKIP = {"Windows": ["blog_with_images_ipynb"]}

MINIMAL_INTEGRATION_TESTS = {}

MINIMAL_IPYNB_INTEGRATION_TESTS = {
    # "delegation_crewai_ipynb": "examples/delegation_crewai.ipynb",
    # "image_generation_crewai_ipynb": "examples/image_generation_crewai.ipynb",
    # "math_crewai_ipynb": "examples/math_crewai.ipynb",
    # "single_crewai_ipynb": "examples/single_crewai.ipynb",
    # "single_llama_index_ipynb": "examples/single_llama_index.ipynb",
    # "single_openai_tools_react_ipynb": "examples/single_openai_tools_react.ipynb",
}

ALL_INTEGRATION_TESTS = dict(
    **INTEGRATION_TESTS,
    **IPYNB_INTEGRATION_TESTS,
    **MINIMAL_INTEGRATION_TESTS,
    **MINIMAL_IPYNB_INTEGRATION_TESTS
)

DEFAULT_CACHE_DIR = Path(__file__).parent / "itest_cache"
DEFAULT_GOLDEN_DIR = Path(__file__).parent / "itest_golden_data"
TIKTOKEN_CACHE_DIR_NAME = "tiktoken_cache"
TIKTOKEN_CACHE_DIR_ENV_VAR = "TIKTOKEN_CACHE_DIR"


def get_args_parser():
    """Argument parser"""
    parser = argparse.ArgumentParser(
        description="Run integration tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-names",
        type=str,
        choices=ALL_INTEGRATION_TESTS.keys(),
        nargs="+",
        help="Names of the tests to run (leave empty to run all tests)",
        default=None,
    )
    parser.add_argument("--cache-dir", type=str, help="Cache directory", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--minimal-only", default=False, action="store_true", help="Run minimal tests"
    )
    parser.add_argument(
        # added to skip problematic tests on Windows workers in GutHub Actions
        "--os",
        type=str,
        default="Unix",
        help="Target operating system",
    )

    return parser


def run_ipynb(ipynb_path: str, strong_cache: bool = False, cache_sub_dir: str = None) -> str:
    """Run jupiter notebook execution"""
    with open(ipynb_path) as f:
        nb = nbformat.read(f, as_version=4)

    # ipynb import and cache settings
    if cache_sub_dir:
        str_strong_cache = "True" if strong_cache else "False"
        cells = [
            new_code_cell("from dotenv import load_dotenv"),
            new_code_cell("load_dotenv()"),
            new_code_cell(
                "from motleycache.caching import enable_cache, disable_cache, set_cache_location"
            ),
            new_code_cell("from motleycache import set_strong_cache, set_cache_blacklist"),
            new_code_cell("enable_cache()"),
            new_code_cell("set_strong_cache({})".format(str_strong_cache)),
            new_code_cell("set_cache_location(r'{}')".format(cache_sub_dir)),
            new_code_cell(
                "set_cache_blacklist(['*openaipublic.blob.core.windows.net/encodings*'])"
            ),
        ]
        for cell in reversed(cells):
            nb.cells.insert(0, cell)

        # ipynb save final result
        # final_result variable must be present in ipynb and store the result of the execution as a string
        ipynb_result_file_path = os.path.join(cache_sub_dir, "ipynb_result.txt")
        save_result_command = "with open(r'{}', 'w') as f:\n\tf.write(final_result)".format(
            ipynb_result_file_path
        )
        cells = [new_code_cell(save_result_command), new_code_cell("disable_cache()")]
        nb.cells += cells
    else:
        ipynb_result_file_path = None

    ep = ExecutePreprocessor(kernel_name="python3")
    ep.preprocess(nb)

    # create result
    if ipynb_result_file_path is not None:
        if os.path.exists(ipynb_result_file_path):
            with open(ipynb_result_file_path) as f:
                result = f.read()
            os.remove(ipynb_result_file_path)
        else:
            raise IpynbIntegrationTestResultNotFound(ipynb_path, ipynb_result_file_path)
    else:
        result = ""

    return nbformat.writes(nb)


def build_ipynb_integration_tests(is_minimal: bool = False) -> dict:
    """Build and return dict of ipynb integration tests functions"""
    test_functions = {}
    tests = MINIMAL_IPYNB_INTEGRATION_TESTS if is_minimal else IPYNB_INTEGRATION_TESTS
    for test_name, nb_path in tests.items():
        if not os.path.exists(nb_path):
            logger.info("Ipynb test notebook {} not found".format(test_name))
            continue

        test_functions[test_name] = partial(run_ipynb, nb_path)

    return test_functions


def run_integration_tests(
    cache_dir: str,
    test_names: Optional[list[str]] = None,
    minimal_only: bool = False,
    target_os: str = "Unix",
):
    failed_tests = {}

    if minimal_only:
        integration_tests = {}
    else:
        integration_tests = copy(INTEGRATION_TESTS)
        integration_tests.update(build_ipynb_integration_tests())

    minimal_integration_tests = copy(MINIMAL_INTEGRATION_TESTS)
    minimal_integration_tests.update(build_ipynb_integration_tests(is_minimal=True))

    for test_key, test_value in minimal_integration_tests.items():
        integration_test_key = "minimal_{}".format(test_key)
        integration_tests[integration_test_key] = test_value

    for current_test_name, test_fn in integration_tests.items():
        if test_names and current_test_name not in test_names:
            continue

        if (
            target_os in INTEGRATION_TESTS_TO_SKIP
            and current_test_name in INTEGRATION_TESTS_TO_SKIP[target_os]
        ):
            logger.info("Skipping test %s for target platform %s", current_test_name, target_os)
            continue

        logger.info("Running test: %s", current_test_name)

        cache_sub_dir = os.path.join(cache_dir, current_test_name)
        set_cache_location(cache_sub_dir)

        if current_test_name in IPYNB_INTEGRATION_TESTS:
            test_fn_kwargs = {
                "cache_sub_dir": cache_sub_dir,
            }
        else:
            test_fn_kwargs = {}

        try:
            test_fn(**test_fn_kwargs)
        except BaseException as e:
            logger.error("Test %s failed: %s", current_test_name, str(e))
            failed_tests[current_test_name] = traceback.format_exc()

    for t, exception in failed_tests.items():
        logger.error("Test %s failed", t)
        logger.error(exception)

    if failed_tests:
        raise IntegrationTestException(test_names=list(failed_tests.keys()))

    logger.info("All tests passed!")


def main():
    configure_logging(verbose=True)
    load_dotenv()

    parser = get_args_parser()
    args = parser.parse_args()

    run_integration_tests(
        cache_dir=args.cache_dir,
        test_names=args.test_names,
        minimal_only=args.minimal_only,
        target_os=args.os,
    )


if __name__ == "__main__":
    main()
