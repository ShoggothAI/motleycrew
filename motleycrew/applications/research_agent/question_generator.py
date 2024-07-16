from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.tools import Tool

from motleycrew.applications.research_agent.question import Question
from motleycrew.common import LLMFramework
from motleycrew.common import logger
from motleycrew.common.llms import init_llm
from motleycrew.common.utils import print_passthrough
from motleycrew.storage import MotleyGraphStore
from motleycrew.tools import MotleyTool

IS_SUBQUESTION_PREDICATE = "is_subquestion"

default_prompt = PromptTemplate.from_template(
    """
You are a part of a team. The ultimate goal of your team is to
answer the following Question: '{question}'.\n
Your team has discovered some new text (delimited by ```) that may be relevant to your ultimate goal.
text: \n ``` {context} ``` \n
Your task is to ask new questions that may help your team achieve the ultimate goal.
If you think that the text is relevant to your ultimate goal, then ask new questions.
New questions should be based only on the text and the goal Question and no other previous knowledge.

You can ask up to {num_questions} new questions.
Return the questions each on a new line and ending with a single question mark.
Don't return anything else except these questions.
"""
)


class QuestionGeneratorTool(MotleyTool):
    """
    Gets a question as input
    Retrieves relevant docs (llama index basic RAG)
    (Retrieves existing questions from graph (to avoid overlap))
    Generates extra questions (research agent prompt)

    Adds questions as children of current q by calling Q insertion tool once
    exits
    """

    def __init__(
        self,
        query_tool: MotleyTool,
        graph: MotleyGraphStore,
        max_questions: int = 3,
        llm: Optional[BaseLanguageModel] = None,
        prompt: str | BasePromptTemplate = None,
    ):
        langchain_tool = create_question_generator_langchain_tool(
            query_tool=query_tool,
            graph=graph,
            max_questions=max_questions,
            llm=llm,
            prompt=prompt,
        )

        super().__init__(langchain_tool)


class QuestionGeneratorToolInput(BaseModel, arbitrary_types_allowed=True):
    """Input for the Question Generator Tool."""

    question: Question = Field(description="The input question for which to generate subquestions.")


def create_question_generator_langchain_tool(
    query_tool: MotleyTool,
    graph: MotleyGraphStore,
    max_questions: int = 3,
    llm: Optional[BaseLanguageModel] = None,
    prompt: str | BasePromptTemplate = None,
):
    if llm is None:
        llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

    llm.bind(json_mode=True)

    if prompt is None:
        prompt = default_prompt
    elif isinstance(prompt, str):
        prompt = PromptTemplate.from_template(prompt)

    assert isinstance(prompt, BasePromptTemplate), "Prompt must be a string or a BasePromptTemplate"

    def insert_questions(input_dict) -> None:
        questions_raw = input_dict["subquestions"].content
        questions = [q.strip() for q in questions_raw.split("\n") if len(q.strip()) > 1]
        for q in questions:
            logger.info("Inserting question: %s", q)
            subquestion = graph.insert_node(Question(question=q))
            graph.create_relation(input_dict["question"], subquestion, IS_SUBQUESTION_PREDICATE)
        logger.info("Inserted %s questions", len(questions))

    def set_context(input_dict: dict):
        node = input_dict["question"]
        node.context = input_dict["context"]

    pipeline = (
        RunnableLambda(print_passthrough)
        | RunnablePassthrough().assign(context=query_tool.to_langchain_tool())
        | RunnableLambda(print_passthrough)
        | RunnablePassthrough().assign(
            subquestions=prompt.partial(num_questions=str(max_questions)) | llm
        )
        | RunnableLambda(print_passthrough)
        | {
            "set_context": RunnableLambda(set_context),
            "insert_questions": RunnableLambda(insert_questions),
        }
    )

    return Tool.from_function(
        func=lambda q: pipeline.invoke({"question": q}),
        name="Question Generator Tool",
        description="""Generate a list of questions based on the input question, 
    and insert them into the knowledge graph.""",
        args_schema=QuestionGeneratorToolInput,
    )
