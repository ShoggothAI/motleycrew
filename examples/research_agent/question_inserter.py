from typing import List

from pathlib import Path

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool

from motleycrew.storage import MotleyGraphStore
from motleycrew.tool import MotleyTool

from question_struct import Question


IS_SUBQUESTION_PREDICATE = "is_subquestion"


class QuestionInsertionTool(MotleyTool):
    def __init__(self, question: Question, graph: MotleyGraphStore):

        langchain_tool = create_question_insertion_langchain_tool(
            name="Question Insertion Tool",
            description="Insert a list of questions (supplied as a list of strings) into the graph.",
            question=question,
            graph=graph,
        )

        super().__init__(langchain_tool)


class QuestionInsertionToolInput(BaseModel):
    """Subquestions of the current question, to be inserted into the knowledge graph."""

    questions: List[str] = Field(description="List of questions to be inserted into the knowledge graph.")


def create_question_insertion_langchain_tool(
    name: str,
    description: str,
    question: Question,
    graph: MotleyGraphStore,
):
    def insert_questions(questions: list[str]) -> None:
        for subquestion in questions:
            subquestion_data = graph.create_entity(Question(question=subquestion).serialize())
            subquestion_obj = Question.deserialize(subquestion_data)
            graph.create_rel(from_id=question.id, to_id=subquestion_obj.id, predicate=IS_SUBQUESTION_PREDICATE)

    return Tool.from_function(
        func=insert_questions,
        name=name,
        description=description,
        args_schema=QuestionInsertionToolInput,
    )


if __name__ == "__main__":
    import kuzu
    from motleycrew.storage import MotleyKuzuGraphStore

    here = Path(__file__).parent
    db_path = here / "test1"
    db = kuzu.Database(db_path)
    graph_store = MotleyKuzuGraphStore(
        db, node_table_schema={"question": "STRING", "answer": "STRING", "context": "STRING"}
    )

    question_data = graph_store.create_entity(Question(question="What is the capital of France?").serialize())
    question = Question.deserialize(question_data)

    children = ["What is the capital of France?", "What is the capital of Germany?"]
    tool = QuestionInsertionTool(question=question, graph=graph_store)
    tool.invoke({"questions": children})

    print(f"docker run -p 8000:8000  -v {db_path}:/database --rm kuzudb/explorer: latest")
    print("MATCH (A)-[r]->(B) RETURN *;")
