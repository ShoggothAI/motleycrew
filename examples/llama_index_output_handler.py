from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun


from motleycrew import MotleyCrew
from motleycrew.agents.llama_index import ReActLlamaIndexMotleyAgent
from motleycrew.common import configure_logging
from motleycrew.tasks import SimpleTask
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.common import AsyncBackend

from langchain_core.tools import StructuredTool


def main():
    """Main function of running the example."""
    search_tool = DuckDuckGoSearchRun()

    def check_output(output: str):
        if "medicine" not in output.lower():
            raise InvalidOutput(
                "Add more information about AI applications in medicine."
            )

        return {"checked_output": output}

    output_handler = StructuredTool.from_function(
        name="output_handler",
        description="Output handler",
        func=check_output,
    )

    # TODO: add LlamaIndex native tools
    researcher = ReActLlamaIndexMotleyAgent(
        prompt_prefix="Your goal is to uncover cutting-edge developments in AI and data science",
        tools=[search_tool],
        output_handler=output_handler,
        verbose=True,
        max_iterations=16,  # default is 10, we add more because the output handler may reject the output
    )

    crew = MotleyCrew(async_backend=AsyncBackend.NONE)

    # Create tasks for your agents
    task = SimpleTask(
        crew=crew,
        name="produce comprehensive analysis report on AI advancements",
        description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
      Identify key trends, breakthrough technologies, and potential industry impacts.
      Your final answer MUST be a full analysis report""",
        agent=researcher,
    )

    # Get your crew to work!
    crew.run()

    print(task.output)
    return task.output


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
