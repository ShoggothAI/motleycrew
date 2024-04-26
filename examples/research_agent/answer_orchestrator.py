from llama_index.core.graph_stores.types import GraphStore

import logging
import sys
import kuzu

from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from motleycrew import MotleyTool
from motleycrew.storage import MotleyGraphStore
from motleycrew.tool.llm_tool import LLMTool

from question_struct import Question
from question_generator import QuestionGeneratorTool
from question_generator import QuestionGeneratorToolInput
from question_answerer import AnswerSubQuestionTool

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class AnswerOrchestrator:
    def __init__(self, storage: MotleyGraphStore, query_tool: MotleyTool):
        self.storage = storage
        self.query_tool = query_tool
        self.question_answering_tool = AnswerSubQuestionTool(graph=self.storage)

    def get_unanswered_available_questions(self) -> list[Question]:
        query = "MATCH (n1:{}) WHERE n1.answer IS NULL AND NOT (n1)-[:{}]->(:{}) RETURN n1;".format(
            self.storage.node_table_name, self.storage.rel_table_name, self.storage.node_table_name
        )

        query_result = self.storage.run_query(query)
        return [Question.deserialize(row[0]) for row in query_result]

    def __call__(self):
        last_question = None

        while True:
            questions = self.storage.get_unanswered_available_questions()
            if not len(questions):
                return last_question
            else:
                last_question = questions[0]
                answerer.invoke({"question": last_question})
        return graph.retrieve(last_question.id)


if __name__ == "__main__":
    from pathlib import Path
    import shutil
    from dotenv import load_dotenv
    from motleycrew.storage import MotleyKuzuGraphStore

    load_dotenv()
    here = Path(__file__).parent
    db_path = here / "research_db"
    shutil.rmtree(db_path, ignore_errors=True)

    db = kuzu.Database(db_path)
    storage = MotleyKuzuGraphStore(
        db, node_table_schema={"question": "STRING", "answer": "STRING", "context": "STRING"}
    )

    query_tool = MotleyTool.from_langchain_tool(
        Tool.from_function(
            func=lambda question: [
                "Germany has consisted of many different states over the years",
                "The capital of France has moved in 1815, from Lyons to Paris",
                "France actually has two capitals, one in the north and one in the south",
            ],
            name="Query Tool",
            description="Query the library for relevant information.",
            args_schema=QuestionGeneratorToolInput,
        )
    )

    orchestrator = KnowledgeGainingOrchestrator(storage=storage, query_tool=query_tool)
    orchestrator(query="Why did Arjuna kill his step-brother?", max_iter=5)


def answer_orchestrator(graph: GraphStore):
    last_question = None
    answerer = AnswerSubQuestionTool(graph=graph)
    while True:
        questions = graph.get_unanswered_available_questions()
        if not len(questions):
            return last_question
        else:
            last_question = questions[0]
            answerer.invoke({"question": last_question})
    return graph.retrieve(last_question.id)
