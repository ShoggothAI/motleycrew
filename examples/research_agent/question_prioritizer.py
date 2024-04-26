from langchain.prompts import PromptTemplate

from motleycrew.tool import LLMTool

PROMPT_TEMPLATE = PromptTemplate(
    template=(
        "You are provided with the following list of questions:"
        " {unanswered_questions} \n"
        " Your task is to choose one question from the above list"
        " that is the most pertinent to the following query:\n"
        " '{original_question}' \n"
        " Respond with one question out of the provided list of questions."
        " Return the questions as it is without any edits."
        " Format your response like:\n"
        " #. question"
    ),
    input_variables=["unanswered_questions", "original_question"],
)


class QuestionPrioritizerTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="Question prioritizer",
            description="Takes the original question and a list of derived questions, "
            "and selects from the latter the one most pertinent to the former",
            prompt=PROMPT_TEMPLATE,
        )


if __name__ == "__main__":
    q = "What color is the sky?"
    unanswered = ["What time of day is it?", "Who was H.P.Lovecraft?"]
    out = QuestionPrioritizerTool().invoke({"unanswered_questions": str(unanswered), "original_question": q})
    print(out)
    print("yay!")
