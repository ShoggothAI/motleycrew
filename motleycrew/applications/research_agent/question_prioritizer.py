from langchain.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    chain,
)
from langchain_core.tools import StructuredTool

from motleycrew.applications.research_agent.question import Question
from motleycrew.common.utils import print_passthrough
from motleycrew.tools import LLMTool
from motleycrew.tools import MotleyTool


class QuestionPrioritizerTool(MotleyTool):
    """Tool to prioritize subquestions based on the original question."""

    def __init__(
        self,
        prompt: str | BasePromptTemplate = None,
    ):
        langchain_tool = create_question_prioritizer_langchain_tool(prompt=prompt)

        super().__init__(langchain_tool)


_default_prompt = PromptTemplate(
    template=(
        "You are provided with the following list of questions:"
        " {unanswered_questions_text} \n"
        " Your task is to choose one question from the above list"
        " that is the most pertinent to the following query:\n"
        " '{original_question_text}' \n"
        " Respond with a single number the chosen question out of the provided list of questions."
        " Return only the number as it is without any edits."
    ),
    input_variables=["unanswered_questions", "original_question"],
)


class QuestionPrioritizerInput(BaseModel, arbitrary_types_allowed=True):
    """Input for the QuestionPrioritizerTool."""

    original_question: Question = Field(description="The original question.")
    unanswered_questions: list[Question] = Field(
        description="Questions to pick the most pertinent to the original question from.",
    )


def create_question_prioritizer_langchain_tool(
    prompt: str | BasePromptTemplate = None,
) -> StructuredTool:
    if prompt is None:
        prompt = _default_prompt

    question_prioritizer = LLMTool(
        prompt=prompt,
        name="Question prioritizer",
        description="Takes the original question and a list of derived questions, "
        "and selects from the latter the one most pertinent to the former",
    )

    @chain
    def get_original_question_text(input_dict: dict):
        return input_dict["original_question"].question

    @chain
    def format_unanswered_questions(input_dict: dict):
        unanswered_questions: list[Question] = input_dict["unanswered_questions"]
        return "\n".join(
            "{}. {}".format(i + 1, question.question)
            for i, question in enumerate(unanswered_questions)
        )

    @chain
    def get_most_pertinent_question(input_dict: dict):
        unanswered_questions: list[Question] = input_dict["unanswered_questions"]
        most_pertinent_question_id = (
            int(input_dict["most_pertinent_question_id_message"].content.strip(" \n.")) - 1
        )
        assert most_pertinent_question_id < len(unanswered_questions)
        return unanswered_questions[most_pertinent_question_id]

    this_chain = (
        RunnablePassthrough.assign(
            original_question_text=lambda x: x["original_question"].question,
            unanswered_questions_text=format_unanswered_questions,
        )
        | RunnableLambda(print_passthrough)
        | RunnablePassthrough.assign(
            most_pertinent_question_id_message=question_prioritizer.to_langchain_tool()
        )
        | RunnableLambda(print_passthrough)
        | get_most_pertinent_question
    )

    langchain_tool = StructuredTool.from_function(
        func=lambda original_question, unanswered_questions: this_chain.invoke(
            {"original_question": original_question, "unanswered_questions": unanswered_questions}
        ),
        name=question_prioritizer.name,
        description=question_prioritizer.tool.description,
        args_schema=QuestionPrioritizerInput,
    )

    return langchain_tool
