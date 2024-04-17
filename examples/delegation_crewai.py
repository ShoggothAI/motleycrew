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

writer = CrewAIMotleyAgent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
You transform complex concepts into compelling narratives.""",
    verbose=True,
    delegation=True,
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
    documents=["paper1.pdf", "paper2.pdf"],  # will be ignored for now
)

task2 = Task(
    crew=crew,
    name="provide a literature summary of recent papers on AI",
    description="""Conduct a comprehensive literature review of the latest advancements in AI in 2024.
Identify key papers, researchers, and companies in the space.
Your final answer MUST be a full literature review with citations""",
    agent=researcher,
)


task3 = Task(
    crew=crew,
    name="produce blog post on AI advancements",
    description="""Using the insights provided by a thorough web search, develop an engaging blog
post that highlights the most significant AI advancements.
Your post should be informative yet accessible, catering to a tech-savvy audience.
Make it sound cool, avoid complex words so it doesn't sound like AI.
Create a blog post of at least 4 paragraphs.""",
    agent=writer,
)

[task1, task2] >> task3

# Get your crew to work!
result = crew.run(
    agents=[researcher, writer],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

print(list(result._done)[0].outputs)
