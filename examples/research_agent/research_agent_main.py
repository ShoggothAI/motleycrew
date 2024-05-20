from pathlib import Path
import shutil
import os
import logging
import platform

import kuzu
from dotenv import load_dotenv

from motleycrew import MotleyCrew
from motleycrew.storage import MotleyKuzuGraphStore
from motleycrew.common.utils import configure_logging
from motleycrew.applications.research_agent.question_task_recipe import QuestionTaskRecipe
from motleycrew.applications.research_agent.answer_task_recipe import AnswerTaskRecipe

from retriever_tool import make_retriever_tool


WORKING_DIR = Path(__file__).parent
if "Dropbox" in WORKING_DIR.parts and platform.system() == "Windows":
    # On Windows, kuzu has file locking issues with Dropbox
    DB_PATH = os.path.realpath(os.path.expanduser("~") + "/Documents/research_db")
else:
    DB_PATH = os.path.realpath(WORKING_DIR / "research_db")

DATA_DIR = os.path.realpath(os.path.join(WORKING_DIR, "../../../mahabharata/text/TinyTales"))
PERSIST_DIR = WORKING_DIR / "storage"

QUESTION = "Why did Arjuna kill Karna, his half-brother?"
MAX_ITER = 2
ANSWER_LENGTH = 200


def main():
    load_dotenv()
    configure_logging(verbose=True)

    shutil.rmtree(DB_PATH)

    query_tool = make_retriever_tool(DATA_DIR, PERSIST_DIR, return_strings_only=True)

    db = kuzu.Database(DB_PATH)
    graph_store = MotleyKuzuGraphStore(db)

    crew = MotleyCrew(graph_store=graph_store)

    question_recipe = QuestionTaskRecipe(
        crew=crew, question=QUESTION, query_tool=query_tool, max_iter=MAX_ITER
    )
    answer_recipe = AnswerTaskRecipe(answer_length=ANSWER_LENGTH, crew=crew)

    question_recipe >> answer_recipe

    done_tasks = crew.run()

    final_answer = done_tasks[-1].question

    print("Question: ", final_answer.question)
    print("Answer: ", final_answer.answer)
    print()
    print("To explore the graph:")
    print(f"docker run -p 8000:8000  -v {DB_PATH}:/database --rm kuzudb/explorer:latest")
    print("MATCH (A)-[r]->(B) RETURN *;")


if __name__ == "__main__":
    main()
