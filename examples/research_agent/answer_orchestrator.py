import logging
import kuzu

from motleycrew.storage import MotleyGraphStore

from question_struct import Question
from question_answerer import AnswerSubQuestionTool


class AnswerOrchestrator:
    def __init__(self, storage: MotleyGraphStore, answer_length: int):
        self.storage = storage
        self.question_answering_tool = AnswerSubQuestionTool(graph=self.storage, answer_length=answer_length)

    def get_unanswered_available_questions(self) -> list[Question]:
        query = (
            "MATCH (n1:{}) "
            "WHERE n1.answer IS NULL AND n1.context IS NOT NULL "
            "AND NOT EXISTS {{MATCH (n1)-[]->(n2:{}) "
            "WHERE n2.answer IS NULL AND n2.context IS NOT NULL}} "
            "RETURN n1"
        ).format(self.storage.node_table_name, self.storage.node_table_name)

        query_result = self.storage.run_cypher_query(query)
        return [Question.deserialize(row[0]) for row in query_result]

    def __call__(self) -> Question | None:
        last_question = None

        while True:
            questions = self.get_unanswered_available_questions()
            logging.info("Available questions: %s", questions)

            if not len(questions):
                logging.info("All questions answered!")
                break
            else:
                last_question = questions[0]
                logging.info("Running answerer for question %s", last_question)
                self.question_answering_tool.invoke({"question": last_question})

        if not last_question:
            logging.warning("Nothing to answer!")
            return

        return Question.deserialize(self.storage.get_entity(last_question.id))


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    from motleycrew.storage import MotleyKuzuGraphStore
    from motleycrew.common.utils import configure_logging

    load_dotenv()
    configure_logging(verbose=True)

    here = Path(__file__).parent
    db_path = here / "research_db"

    db = kuzu.Database(db_path)
    storage = MotleyKuzuGraphStore(
        db, node_table_schema={"question": "STRING", "answer": "STRING", "context": "STRING"}
    )

    orchestrator = AnswerOrchestrator(storage=storage, answer_length=30)
    result = orchestrator()
    print(result)
