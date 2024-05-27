from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew import MotleyCrew
from motleycrew.agents.langchain.openai_tools_react import ReactOpenAIToolsAgent
from motleycrew.agents.langchain.react import ReactMotleyAgent
from motleycrew.common import configure_logging
from motleycrew.tasks import SimpleTask
from motleycache import enable_cache


def main():
    search_tool = DuckDuckGoSearchRun()

    tools = [search_tool]

    researcher = ReactOpenAIToolsAgent(tools=tools, verbose=True)
    researcher2 = ReactMotleyAgent(tools=tools, verbose=True)

    outputs = []

    for r in [researcher, researcher2]:
        crew = MotleyCrew()
        task = SimpleTask(
            crew=crew,
            name="produce comprehensive analysis report on AI advancements",
            description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
          Identify key trends, breakthrough technologies, and potential industry impacts.
          Your final answer MUST be a full analysis report""",
            agent=r,
        )
        result = crew.run()

        # Get your crew to work!
        print(task.output)
        outputs.append(task.output)

    return outputs


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
