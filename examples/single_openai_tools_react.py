from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew import MotleyCrew
from motleycrew.agent.langchain.openai_tools_react import ReactOpenAIToolsAgent
from motleycrew.agent.langchain.react import ReactMotleyAgent
from motleycrew.common.utils import configure_logging
from motleycrew.caching import enable_cache


def main():
    search_tool = DuckDuckGoSearchRun()

    tools = [search_tool]

    researcher = ReactOpenAIToolsAgent(tools=tools, verbose=True)
    researcher2 = ReactMotleyAgent(tools=tools, verbose=True)

    outputs = []

    for r in [researcher, researcher2]:
        crew = MotleyCrew()
        task = crew.create_simple_task(
            name="produce comprehensive analysis report on AI advancements",
            description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
          Identify key trends, breakthrough technologies, and potential industry impacts.
          Your final answer MUST be a full analysis report""",
            agent=r,
        )
        result = crew.run(
            verbose=2,  # You can set it to 1 or 2 to different logging levels
        )

        # Get your crew to work!
        print(task.output)
        outputs.append(task.output)

    return outputs


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
