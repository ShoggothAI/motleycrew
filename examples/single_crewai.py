from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew import MotleyCrew, Task
from motleycrew.agent.crewai import CrewAIMotleyAgent

load_dotenv()

search_tool = DuckDuckGoSearchRun()

researcher = CrewAIMotleyAgent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and data science",
    backstory="""You work at a leading tech think tank.
Your expertise lies in identifying emerging trends.
You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    delegation=False,
    tools=[search_tool],
)


# Create tasks for your agents
crew = MotleyCrew()
task1 = Task(
    crew=crew,
    name="produce comprehensive analysis report on AI advancements",
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Your final answer MUST be a full analysis report""",
    agent=researcher,
    documents=["paper1.pdf", "paper2.pdf"],
)

# Instantiate your crew with a sequential process
result = crew.run(
    agents=[researcher],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
print(list(result._done)[0].outputs)

print("######################")
