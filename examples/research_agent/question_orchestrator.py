import logging
import sys
import kuzu

from langchain.tools import Tool

from motleycrew import MotleyTool
from motleycrew.storage import MotleyGraphStore

from question_struct import Question
from question_generator import QuestionGeneratorTool
from question_generator import QuestionGeneratorToolInput
from question_prioritizer import QuestionPrioritizerTool

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class KnowledgeGainingOrchestrator:
    def __init__(self, storage: MotleyGraphStore, query_tool: MotleyTool):
        self.storage = storage
        self.query_tool = query_tool
        self.question_prioritization_tool = QuestionPrioritizerTool()
        self.question_generation_tool = QuestionGeneratorTool(query_tool=query_tool, graph=self.storage)

    def get_unanswered_questions(self, only_without_children: bool = False) -> list[Question]:
        if only_without_children:
            query = "MATCH (n1:{}) WHERE n1.answer IS NULL AND NOT (n1)-[:{}]->(:{}) RETURN n1;".format(
                self.storage.node_table_name, self.storage.rel_table_name, self.storage.node_table_name
            )
        else:
            query = "MATCH (n1:{}) WHERE n1.answer IS NULL RETURN n1;".format(self.storage.node_table_name)

        query_result = self.storage.run_query(query)
        return [Question.deserialize(row[0]) for row in query_result]

    def __call__(self, query: str, max_iter: int):
        self.storage.create_entity({"question": query})

        for iter_n in range(max_iter):
            logging.info("====== Iteration %s of %s ======", iter_n, max_iter)

            unanswered_questions = self.get_unanswered_questions(only_without_children=True)
            logging.info("Loaded unanswered questions: %s", unanswered_questions)

            question_prioritization_tool_input = {
                "unanswered_questions": "\n".join(
                    f"{i}. {question.question}" for i, question in enumerate(unanswered_questions)
                ),
                "original_question": query,
            }
            most_pertinent_question_raw = self.question_prioritization_tool.invoke(
                question_prioritization_tool_input
            ).content
            logging.info("Most pertinent question according to the tool: %s", most_pertinent_question_raw)

            i, most_pertinent_question_text = most_pertinent_question_raw.split(".", 1)
            i = int(i)
            assert i < len(unanswered_questions)

            most_pertinent_question = unanswered_questions[i]
            assert most_pertinent_question_text.strip() == most_pertinent_question.question.strip()

            logging.info("Generating new questions")
            self.question_generation_tool.invoke({"question": most_pertinent_question})


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
    orchestrator(query="What is the capital of France?", max_iter=5)
