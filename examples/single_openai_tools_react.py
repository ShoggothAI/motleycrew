from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew import MotleyCrew, Task, MotleyTool
from motleycrew.agent.langchain.openai_tools_react import ReactOpenAIToolsAgent
from motleycrew.agent.langchain.react import ReactMotleyAgent

# # You can delete this block if you don't want to use Langsmith
# from langsmith import Client
#
# unique_id = uuid4().hex[0:8]
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = credentials["LANGCHAIN_API_KEY"]
#
# client = Client()
# # End of Langsmith block

load_dotenv()

search_tool = DuckDuckGoSearchRun()


tools = [search_tool]


researcher = ReactOpenAIToolsAgent(tools=tools, verbose=True)
researcher2 = ReactMotleyAgent(tools=tools, verbose=True)

for r in [researcher, researcher2]:

    crew = MotleyCrew()
    task1 = Task(
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
    print(list(result._done)[0].outputs)

print("######################")
