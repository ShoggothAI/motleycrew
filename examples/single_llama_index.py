from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun


from motleycrew import MotleyCrew, Task
from motleycrew.agent.llama_index import ReActLlamaIndexMotleyAgent
from motleycrew.common.utils import configure_logging


def main():
    """Main function of running the example."""
    search_tool = DuckDuckGoSearchRun()

    # TODO: add LlamaIndex native tools
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

    # Get your crew to work!
    result = crew.run(
        agents=[researcher],
        verbose=2,  # You can set it to 1 or 2 to different logging levels
    )

    outputs = task1.outputs
    print(outputs)

    return [output.response for output in outputs]


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
