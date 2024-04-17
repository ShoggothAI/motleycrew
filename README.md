# motleycrew
A lightweight agent interaction framework.

Minimal example with maximum automation (might take a while to build ;) ):

```python
from motleycrew import Task, MotleyCrew

task = Task("""Research arxiv on the latest trends in machine learning
and produce an engaging blog post on the topic""",
            documents=["paper1.pdf", "paper2.pdf"])
crew = MotleyCrew(tasks=[task])
crew.run()
```
Come to think of it, might it make sense to make it 2 libraries, one with the interaction
primitives, and the other on top of it with the automation?

Here the MotleyCrew auto-spawns agents to complete the task, and picks additional tools for them.

Short term, more crewAI-style (here some is copy-pasted from crewAI,
will need to edit before going public)
```python
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

from motley_crew import Task, MotleyCrew
from motley_crew.agents import LangchainAgent, MotleyAgent

tools=[DuckDuckGoSearchRun()]
researcher_prompt = hub.pull("hwchase17/openai-tools-agent")
llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)

researcher_agent = AgentExecutor(
    agent=create_openai_tools_agent(llm, tools, researcher_prompt),
    tools=tools,
    verbose=True,
)

researcher = LangchainAgent(
    agent=researcher_agent,
    goal="Research the web and any documents they are given, and summarize the results",
    allow_delegation=False
)

writer = MotleyAgent(
    goal="Craft compelling content on tech advancements",
    description="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
    verbose=True,
    kind = "crewai",
    delegation=True,
)

# Create tasks for your agents
task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Your final answer MUST be a full analysis report""",
    agent=researcher,
    documents = ["paper1.pdf", "paper2.pdf"],
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.
  """,
    agent=writer,
    depends_on=task1,
)

# Instantiate your crew with a sequential process
crew = MotleyCrew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.run()
