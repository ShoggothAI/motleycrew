from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew import MotleyCrew, Task
from motleycrew.agent.langchain.openai_tools_react import ReactOpenAIToolsAgent
from motleycrew.agent.langchain.react import ReactMotleyAgent
from motleycrew.common.utils import configure_logging


def main():
    search_tool = DuckDuckGoSearchRun()

    tools = [search_tool]

    researcher = ReactOpenAIToolsAgent(tools=tools, verbose=True)
    researcher2 = ReactMotleyAgent(tools=tools, verbose=True)

    outputs = []

    for r in [researcher, researcher2]:
        crew = MotleyCrew()
        task = Task(
            crew=crew,
            name="produce comprehensive analysis report on AI advancements",
            description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
          Identify key trends, breakthrough technologies, and potential industry impacts.
          Your final answer MUST be a full analysis report""",
            agent=r,
            documents=["paper1.pdf", "paper2.pdf"],  # will be ignored for now
        )
        result = crew.run(
            agents=[r],
            verbose=2,  # You can set it to 1 or 2 to different logging levels
        )

        # Get your crew to work!
        print(task.outputs)
        outputs += task.outputs

    return outputs


if __name__ == "__main__":
    from motleycrew.caching import enable_cache, set_cache_location, set_strong_cache

    configure_logging(verbose=True)

    load_dotenv()
    enable_cache()
    set_strong_cache(False)
    main()
