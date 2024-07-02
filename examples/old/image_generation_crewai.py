from dotenv import load_dotenv

from motleycrew import MotleyCrew
from motleycrew.agents.crewai import CrewAIMotleyAgent
from motleycrew.tools.image.dall_e import DallEImageGeneratorTool
from motleycrew.common import configure_logging


def main():
    """Main function of running the example."""
    image_generator_tool = DallEImageGeneratorTool()
    # For saving images locally use the line below
    # image_generator_tool = DallEImageGeneratorTool(images_directory="images")

    illustrator = CrewAIMotleyAgent(
        role="Illustrator",
        goal="Create beautiful and insightful illustrations",
        prompt_prefix="You are an expert in creating illustrations for all sorts of concepts and articles. "
        "You do it by skillfully prompting a text-to-image model.\n"
        "Your final answer MUST be the exact URL or filename of the illustration.",
        backstory="",
        verbose=True,
        delegation=False,
        tools=[image_generator_tool],
    )

    writer = CrewAIMotleyAgent(
        role="Short stories writer",
        goal="Write short stories for children",
        prompt_prefix="You are an accomplished children's writer, known for your funny and interesting short stories.\n"
        "Many parents around the world read your books to their children.",
        backstory="",
        verbose=True,
        delegation=True,
        tools=[illustrator],
    )

    # Create tasks for your agents
    crew = MotleyCrew()
    write_task = crew.create_simple_task(
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
    crew.run()

    print(write_task.output)
    return write_task.output


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
