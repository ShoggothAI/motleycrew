from typing import Optional, List, Tuple


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.tools import StructuredTool, Tool
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    chain,
)

from llama_index.core.graph_stores.types import GraphStore
from motleycrew.tool import MotleyTool, LLMTool
from motleycrew.common.utils import print_passthrough

_default_prompt = PromptTemplate.from_template(
    """
    You are a research agent who answers complex questions with clear, crisp and detailed answers.
     You are provided with a question and some research notes prepared by your team.
     Question: {question} \n
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
    def __init__(
        self,
        graph: GraphStore,
        prompt: str | BasePromptTemplate = None,
    ):
        langchain_tool = create_answer_question_langchain_tool(
            graph=graph,
            prompt=prompt,
        )

        super().__init__(langchain_tool)


class QuestionAnswererInput(BaseModel):
    """Data on the question to answer."""

    question_id: int = Field(
        description="Id of the question node to process.",
    )
    notes: str = Field(
        description="The notes that contain the sub-questions and their answers.",
    )
    question: str = Field(
        description="The question to answer.",
    )


def create_answer_question_langchain_tool(
    graph: GraphStore,
    prompt: str | BasePromptTemplate = None,
) -> Tool:
    """
    Creates a LangChainTool for the AnswerSubQuestionTool.
    """
    if prompt is None:
        prompt = _default_prompt

    subquestion_answerer = LLMTool(
        prompt=prompt,
        name="Question answerer",
        description="Tool to answer a question from notes and sub-questions",
    )
    """
    	Gets a valid question node ID, question, and context as input dict
    	Retrieves child quuestion answers
    	Feeds all that to LLM to answer Q (research_agent prompt)
    	Attaches answer to the node
    """

    @chain
    def retrieve_sub_question_answers(**kwargs) -> List[Tuple[str, str]]:
        """
        Retrieves the answers to the sub-questions of a given question.
        """
        sub_questions = graph.get_sub_questions(kwargs["question_id"])
        out = []
        for sq in sub_questions:
            if sq["answer"] is not None:
                out.append((sq["question"], sq["answer"]))
        return out

    @chain
    def merge_notes(**kwargs) -> str:
        """
        Merges the notes and the sub-question answers.
        """
        notes = kwargs["notes"]
        sub_question_answers = kwargs["sub_question_answers"]
        notes += "\n\n"
        for q, a in sub_question_answers:
            notes += f"Q: {q}\nA: {a}\n\n"
        return notes

    @chain
    def insert_answer(answer: str, question_id: int) -> None:
        """
        Inserts the answer into the graph.
        """
        graph.update_properties(id=question_id, values={"answer": answer})

    this_chain = (
        {
            "sub_question_answers": retrieve_sub_question_answers,
            "input": RunnablePassthrough(),
        }
        | merge_notes
        | {
            "answer": subquestion_answerer.to_langchain_tool(),
            "question_id": RunnablePassthrough(),
        }
        | RunnableLambda(print_passthrough)
        | insert_answer
    )

    langchain_tool = Tool.from_function(
        func=this_chain.invoke,
        name="Answer Sub-Question Tool",
        description="Answer a question based on the notes and sub-questions.",
        args_schema=QuestionAnswererInput,
    )

    return langchain_tool
