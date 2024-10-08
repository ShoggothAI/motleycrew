from dotenv import load_dotenv

from motleycrew import MotleyCrew
from motleycrew.agents.crewai import CrewAIMotleyAgent
from motleycrew.common import configure_logging
from motleycrew.tasks import SimpleTask
from motleycrew.tools.code import PythonREPLTool


def main():
    """Main function of running the example."""
    repl_tool = PythonREPLTool()

    # Define your agents with roles and goals
    solver1 = CrewAIMotleyAgent(
        role="High School Math Teacher",
        goal="Generate great solutions to math problems",
        backstory="""You are a high school math teacher with a passion for problem-solving.
    To solve a math problem, you first reason about it, step by step, then generate the code to solve it exactly,
    using sympy, then use the REPL tool to evaluate that code, and then
    use the output to generate a human-readable solution.""",
        verbose=True,
        delegation=False,
        tools=[repl_tool],
    )

    solver = solver1

    problems = [
        "Problem: If $725x + 727y = 1500$ and $729x+ 731y = 1508$, "
        "what are the values of $x$, $y$, and $x - y$ ?",
        "Drei Personen erhalten zusammen Fr. 2450. Die erste Person erh\u00e4lt Fr. 70 mehr als "
        "die zweite, aber Fr. 60 weniger als die dritte.\nWie viel erh\u00e4lt jede Person? ",
        "Find all numbers $a$ for which the graph of $y=x^2+a$ and the graph of $y=ax$ intersect. "
        "Express your answer in interval notation.",
    ]

    # Create tasks for your agents
    crew = MotleyCrew()
    task = SimpleTask(
        crew=crew,
        name="solve math problem",
        description=f"""Create a nice human-readable solution to the following problem:
        {problems[1]}
         IN THE SAME LANGUAGE AS THE PROBLEM""",
        agent=solver,
    )

    # Get your crew to work!
    crew.run()

    print(task.output)
    return task.output


if __name__ == "__main__":
    configure_logging(verbose=True)

    load_dotenv()
    main()
