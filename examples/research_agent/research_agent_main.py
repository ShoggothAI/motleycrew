from pathlib import Path
import shutil
import os
import kuzu
from dotenv import load_dotenv
from motleycrew.storage import MotleyKuzuGraphStore
from motleycrew.common.utils import configure_logging

from question_orchestrator import KnowledgeGainingOrchestrator
from answer_orchestrator import AnswerOrchestrator
from retriever_tool import make_retriever_tool


WORKING_DIR = Path(__file__).parent
DB_PATH = WORKING_DIR / "research_db"
DATA_DIR = os.path.join(WORKING_DIR, "mahabharata/text/TinyTales")
PERSIST_DIR = WORKING_DIR / "storage"

QUESTION = "Why did Arjuna kill Karna, his half-brother?"
MAX_ITER = 2
ANSWER_LENGTH = 200


def main():
    load_dotenv()
    configure_logging(verbose=True)

    shutil.rmtree(DB_PATH, ignore_errors=True)

    db = kuzu.Database(DB_PATH)
    storage = MotleyKuzuGraphStore(
        db, node_table_schema={"question": "STRING", "answer": "STRING", "context": "STRING"}
    )

    query_tool = make_retriever_tool(DATA_DIR, PERSIST_DIR, return_strings_only=True)

    question_orchestrator = KnowledgeGainingOrchestrator(storage=storage, query_tool=query_tool)
    answer_orchestrator = AnswerOrchestrator(storage=storage, answer_length=ANSWER_LENGTH)

    question_orchestrator(query=QUESTION, max_iter=MAX_ITER)
    answered_question = answer_orchestrator()

    print("Question: ", answered_question.question)
    print("Answer: ", answered_question.answer)
    print()
    print("To explore the graph:")
    print(f"docker run -p 8000:8000  -v {DB_PATH}:/database --rm kuzudb/explorer:latest")
    print("MATCH (A)-[r]->(B) RETURN *;")


if __name__ == "__main__":
    main()
