from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from motleycache import enable_cache

from motleycrew import MotleyCrew
from motleycrew.agents import MotleyOutputHandler
from motleycrew.agents.langchain.tool_calling_react import (
    ReActToolCallingMotleyAgent,
)
from motleycrew.common import configure_logging
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.tasks import SimpleTask


def main():
    search_tool = DuckDuckGoSearchRun()

    tools = [search_tool]

    class ReportOutputHandler(MotleyOutputHandler):
        def handle_output(self, output: str):
            if "medicine" not in output.lower():
                raise InvalidOutput("Add more information about AI applications in medicine.")

            if "2024" in self.agent_input["prompt"]:
                output += "\n\nThis report is up-to-date for 2024."

            output += f"\n\nBrought to you by motleycrew's {self.agent}."

            return {"checked_output": output}

    researcher = ReActToolCallingMotleyAgent(
        tools=tools,
        verbose=True,
        chat_history=True,
        output_handler=ReportOutputHandler(),
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
    enable_cache()
    configure_logging(verbose=True)

    load_dotenv()
    main()
