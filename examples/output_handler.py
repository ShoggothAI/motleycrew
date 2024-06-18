from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew import MotleyCrew
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingAgent
from motleycrew.common import configure_logging
from motleycrew.tasks import SimpleTask

from motleycrew.common.exceptions import InvalidOutput


from langchain_core.tools import StructuredTool


def main():
    search_tool = DuckDuckGoSearchRun()

    tools = [search_tool]

    def check_output(output: str):
        if "medicine" not in output.lower():
            raise InvalidOutput("Add more information about AI applications in medicine.")

        return {"checked_output": output}

    output_handler = StructuredTool.from_function(
        name="output_handler",
        description="Output handler",
        func=check_output,
    )

    researcher = ReActToolCallingAgent(
        tools=tools,
        verbose=True,
        chat_history=True,
        output_handler=output_handler,
    )

    crew = MotleyCrew()
    task = SimpleTask(
        crew=crew,
        name="produce comprehensive analysis report on AI advancements in 2024",
        description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
      Identify key trends, breakthrough technologies, and potential industry impacts.
      Your final answer MUST be a full analysis report""",
        agent=researcher,
    )
    crew.run()
    print(task.output)


if __name__ == "__main__":
    from motleycache import enable_cache

    enable_cache()
    configure_logging(verbose=True)

    load_dotenv()
    main()
