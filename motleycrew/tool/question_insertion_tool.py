from typing import List

from pathlib import Path

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool

# TODO: fallback interface if LlamaIndex is not available
from llama_index.core.graph_stores.types import GraphStore

from motleycrew.tool import MotleyTool


class QuestionInsertionTool(MotleyTool):
    def __init__(self, question: str, graph: GraphStore):

        langchain_tool = create_question_insertion_langchain_tool(
            name="Question Insertion Tool",
            description="Insert a list of questions (supplied as a list of strings) into the graph.",
            question=question,
            graph=graph,
        )

        super().__init__(langchain_tool)


class QuestionInsertionToolInput(BaseModel):
    """Subquestions of the current question, to be inserted into the knowledge graph."""

    questions: List[str] = Field(
        description="List of questions to be inserted into the knowledge graph."
    )


def create_question_insertion_langchain_tool(
    name: str,
    description: str,
    question: str,
    graph: GraphStore,
):
    def insert_questions(questions: list[str]) -> None:
        for subquestion in questions:
            # TODO: change! This is a placeholder implementation
            graph.upsert_triplet(question, "IS_SUBQUESTION", subquestion)

    return Tool.from_function(
        func=insert_questions,
        name=name,
        description=description,
        args_schema=QuestionInsertionToolInput,
    )


if __name__ == "__main__":
    import kuzu
    from llama_index.graph_stores.kuzu import KuzuGraphStore

    here = Path(__file__).parent
    db_path = here / "test1"
    db = kuzu.Database(db_path)
    graph_store = KuzuGraphStore(db)

    children_1 = ["What is the capital of France?", "What is the capital of Germany?"]
    children_2 = ["What is the capital of Italy?", "What is the capital of Spain?"]
    tool = QuestionInsertionTool(question="Starting question", graph=graph_store)
    tool.invoke({"questions": children_1})
    tool2 = QuestionInsertionTool(
        question="What is the capital of France?", graph=graph_store
    )
    tool2.invoke({"questions": children_2})
    print(
        f"docker run -p 8000:8000  -v {db_path}:/database --rm kuzudb/explorer: latest"
    )
    print("MATCH (A)-[r]->(B) RETURN *;")
