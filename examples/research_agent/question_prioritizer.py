from langchain.prompts import PromptTemplate

from motleycrew.tool import LLMTool

prompt = PromptTemplate.from_template(
    """You are provided with the following list of questions:
{unanswered_questions} \n
Your task is to choose one question from the above list
that is the most pertinent to the following query:\n
'{original_question}' \n
Respond with one question out of the provided list of questions.
Return the question as it is without any edits."""
)

prioritizer = LLMTool(
    name="Question prioritizer",
    description="""Takes the original question and a list of derived questions, 
and selects from the latter the one mpst pertinent to the former.""",
    prompt=prompt,
)


if __name__ == "__main__":
    q = "What color is the sky?"
    unanswered = ["What time of day is it?", "Who was H.P.Lovecraft?"]
    out = prioritizer.invoke(
        {"unanswered_questions": str(unanswered), "original_question": q}
    )
    print(out)
    print("yay!")
