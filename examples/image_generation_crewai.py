from dotenv import load_dotenv

from motleycrew import MotleyCrew, Task
from motleycrew.agent.crewai import CrewAIMotleyAgent
from motleycrew.tool.image_generation import DallEImageGeneratorTool
from motleycrew.common.utils import configure_logging

load_dotenv()
configure_logging(verbose=True)

image_generator_tool = DallEImageGeneratorTool()
# For saving images locally use the line below
# image_generator_tool = DallEImageGeneratorTool(images_directory="images")

writer = CrewAIMotleyAgent(
    role="Short stories writer",
    goal="Write short stories for children",
    backstory="You are an accomplished children's writer, known for your funny and interesting short stories.\n"
    "Many parents around the world read your books to their children.",
    verbose=True,
    delegation=True,
)

illustrator = CrewAIMotleyAgent(
    role="Illustrator",
    goal="Create beautiful and insightful illustrations",
    backstory="You are an expert in creating illustrations for all sorts of concepts and articles. "
    "You do it by skillfully prompting a text-to-image model.\n"
    "Your final answer MUST be the exact URL or filename of the illustration.",
    verbose=True,
    delegation=False,
    tools=[image_generator_tool],
)

# Create tasks for your agents
crew = MotleyCrew()
write_task = Task(
    crew=crew,
    name="write a short story about a cat",
    description="Creatively write a short story of about 4 paragraphs "
    "about a house cat that was smarter than its owners. \n"
    "Write it in a cool and simple language, "
    "making it intriguing yet suitable for children's comprehension.\n"
    "You must include a fun illustration.\n"
    "Your final answer MUST be the full story with the illustration URL or filename attached.",
    agent=writer,
)


# Get your crew to work!
result = crew.run(
    agents=[illustrator, writer],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

print(write_task.outputs)
