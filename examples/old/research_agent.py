from pathlib import Path
import shutil
import os

import kuzu
from dotenv import load_dotenv

# This assumes you have a .env file in the examples folder, containing your OpenAI key
load_dotenv()

WORKING_DIR = Path(os.path.realpath(".."))

from motleycrew import MotleyCrew
from motleycrew.storage import MotleyKuzuGraphStore
from motleycrew.common import configure_logging
from motleycrew.applications.research_agent.question_task import QuestionTask
from motleycrew.applications.research_agent.answer_task import AnswerTask
from motleycrew.tools.simple_retriever_tool import SimpleRetrieverTool
from motleycrew.common import AsyncBackend
from motleycache import enable_cache

configure_logging(verbose=True)
enable_cache()

DATA_DIR = os.path.realpath(os.path.join(WORKING_DIR, "mahabharata/text/TinyTales"))
PERSIST_DIR = WORKING_DIR / "storage"
query_tool = SimpleRetrieverTool(DATA_DIR, PERSIST_DIR, return_strings_only=True)

# Create a new graph store
shutil.rmtree(PERSIST_DIR / "kuzu_db", ignore_errors=True)
db = kuzu.Database(str(PERSIST_DIR / "kuzu_db"))
graph_store = MotleyKuzuGraphStore(db)

crew = MotleyCrew(async_backend=AsyncBackend.ASYNCIO, graph_store=graph_store)

QUESTION = "Why did Arjuna kill Karna, his half-brother?"
MAX_ITER = 10
ANSWER_LENGTH = 200


# We need to pass the crew to the Tasks so they have access to the graph store
# and the crew is aware of them

# The question task is responsible for new question generation
question_task = QuestionTask(
    crew=crew,
    question=QUESTION,
    query_tool=query_tool,
    max_iter=MAX_ITER,
    allow_async_units=True,
)

# The answer task is responsible for rolling the answers up the tree
answer_task = AnswerTask(answer_length=ANSWER_LENGTH, crew=crew)

# Only kick off the answer task once the question task is done
question_task >> answer_task

# And now run the recipes
done_items = crew.run()

# Print the final answer
question = done_items[-1].question
print(question.question)
print(question.answer)
