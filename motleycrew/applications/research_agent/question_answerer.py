from langchain.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    chain,
)
from langchain_core.tools import Tool

from motleycrew.applications.research_agent.question import Question
from motleycrew.common.utils import print_passthrough
from motleycrew.storage import MotleyGraphStore
from motleycrew.tools import MotleyTool, LLMTool

_default_prompt = PromptTemplate.from_template(
    """
    You are a research agent who answers complex questions with clear, crisp and detailed answers.
     You are provided with a question and some research notes prepared by your team.
     Question: {question_text} \n
     Notes: {notes} \n
     Your task is to answer the question entirely based on the given notes.
     The notes contain a list of intermediate-questions and answers that may be helpful to you in writing an answer.
     Use only the most relevant information from the notes while writing your answer.
     Do not use any prior knowledge while writing your answer, Do not make up the answer.
     If the notes are not relevant to the question, just return 'Context is insufficient to answer the question'.
     Remember your goal is to answer the question as objectively as possible.
     Write your answer succinctly in less than {answer_length} words."""
)


class AnswerSubQuestionTool(MotleyTool):
    """Tool to answer a question based on the notes and sub-questions."""

    def __init__(
        self,
        graph: MotleyGraphStore,
        answer_length: int,
        prompt: str | BasePromptTemplate = None,
    ):
        langchain_tool = create_answer_question_langchain_tool(
            graph=graph,
            answer_length=answer_length,
            prompt=prompt,
        )

        super().__init__(langchain_tool)


class QuestionAnswererInput(BaseModel, arbitrary_types_allowed=True):
    question: Question = Field(
        description="Question node to process.",
    )


def get_subquestions(graph: MotleyGraphStore, question: Question) -> list[Question]:
    query = (
        "MATCH (n1:{})-[]->(n2:{}) "
        "WHERE n1.id = $question_id and n2.context IS NOT NULL "
        "RETURN n2"
    ).format(Question.get_label(), Question.get_label())

    query_result = graph.run_cypher_query(
        query, parameters={"question_id": question.id}, container=Question
    )
    return query_result


def create_answer_question_langchain_tool(
    graph: MotleyGraphStore,
    answer_length: int,
    prompt: str | BasePromptTemplate = None,
) -> Tool:
    if prompt is None:
        prompt = _default_prompt

    subquestion_answerer = LLMTool(
        prompt=prompt.partial(answer_length=str(answer_length)),
        name="Question answerer",
        description="Tool to answer a question from notes and sub-questions",
    )
    """
    Gets a valid question node ID, question, and context as input dict
    Retrieves child question answers
    Feeds all that to LLM to answer Q (research_agent prompt)
    Attaches answer to the node
    """

    @chain
    def write_notes(input_dict: dict) -> str:
        """
        Merges the notes and the sub-question answers.
        """
        question = input_dict["question"]
        subquestions = get_subquestions(graph=graph, question=question)

        notes = "\n".join(question.context)
        notes += "\n\n"
        for question in subquestions:
            notes += f"Q: {question.question}\nA: {question.answer}\n\n"
        return notes

    @chain
    def insert_answer(input_dict: dict) -> None:
        """
        Inserts the answer into the graph.
        """
        question = input_dict["question"]
        answer = input_dict["answer"].content
        question.answer = answer
        return answer

    this_chain = (
        RunnablePassthrough.assign(
            question_text=lambda x: x["question"].question, notes=write_notes
        )
        | RunnableLambda(print_passthrough)
        | RunnablePassthrough.assign(answer=subquestion_answerer.to_langchain_tool())
        | RunnableLambda(print_passthrough)
        | insert_answer
    )

    langchain_tool = Tool.from_function(
        func=lambda q: this_chain.invoke({"question": q}),
        name="Answer Sub-Question Tool",
        description="Answer a question based on the notes and sub-questions.",
        args_schema=QuestionAnswererInput,
    )

    return langchain_tool
