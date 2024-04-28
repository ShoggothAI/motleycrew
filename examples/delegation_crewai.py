from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew import MotleyCrew, TaskRecipe
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
    tools=[search_tool],
    delegation=False,  # Will be ignored
)

writer = CrewAIMotleyAgent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
You transform complex concepts into compelling narratives.""",
    verbose=True,
)

# Create tasks for your agents
crew = MotleyCrew()
task1 = crew.create_task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
Identify key trends, breakthrough technologies, and potential industry impacts.
Your final answer MUST be a full analysis report""",
    agent=researcher,
)

task2 = crew.create_task(
    name="description",
    description="""Conduct a comprehensive literature review of the latest advancements in AI in 2024.
Identify key papers, researchers, and companies in the space.
Your final answer MUST be a full literature review with citations""",
    agent=researcher,
)


task3 = crew.create_task(
    name="generate",
    description="""Using the insights provided by a thorough web search, develop an engaging blog
post that highlights the most significant AI advancements.
Your post should be informative yet accessible, catering to a tech-savvy audience.
Make it sound cool, avoid complex words so it doesn't sound like AI.
Create a blog post of at least 4 paragraphs.""",
    agent=writer,
)

[task1, task2] >> task3

# Get your crew to work!
result = crew.run(verbose=2)

print(list(result._done)[0].outputs)
