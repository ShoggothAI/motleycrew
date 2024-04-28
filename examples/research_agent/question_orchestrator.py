import logging

from motleycrew import MotleyTool
from motleycrew.storage import MotleyGraphStore

from question_struct import Question
from question_generator import QuestionGeneratorTool
from question_prioritizer import QuestionPrioritizerTool


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

        query_result = self.storage.run_cypher_query(query)
        return [Question.deserialize(row[0]) for row in query_result]

    def __call__(self, query: str, max_iter: int):
        question = Question(question=query)
        self.storage.create_entity(question.serialize())

        for iter_n in range(max_iter):
            logging.info("====== Iteration %s of %s ======", iter_n + 1, max_iter)

            unanswered_questions = self.get_unanswered_questions(only_without_children=True)
            logging.info("Loaded unanswered questions: %s", unanswered_questions)

            most_pertinent_question = self.question_prioritization_tool.invoke(
                {"original_question": question, "unanswered_questions": unanswered_questions}
            )
            logging.info("Most pertinent question according to the tool: %s", most_pertinent_question)

            logging.info("Generating new questions")
            self.question_generation_tool.invoke({"question": most_pertinent_question})


if __name__ == "__main__":
    from pathlib import Path
    import shutil
    import os
    import kuzu
    from dotenv import load_dotenv
    from motleycrew.storage import MotleyKuzuGraphStore
    from motleycrew.common.utils import configure_logging

    from retriever_tool import make_retriever_tool

    load_dotenv()
    configure_logging(verbose=True)

    here = Path(__file__).parent
    db_path = here / "research_db"
    shutil.rmtree(db_path, ignore_errors=True)

    db = kuzu.Database(db_path)
    storage = MotleyKuzuGraphStore(
        db, node_table_schema={"question": "STRING", "answer": "STRING", "context": "STRING"}
    )

    DATA_DIR = os.path.join(here, "mahabharata/text/TinyTales")

    PERSIST_DIR = "./storage"

    query_tool = make_retriever_tool(DATA_DIR, PERSIST_DIR, return_strings_only=True)

    orchestrator = KnowledgeGainingOrchestrator(storage=storage, query_tool=query_tool)
    orchestrator(query="Why did Arjuna kill Karna, his half-brother?", max_iter=5)
