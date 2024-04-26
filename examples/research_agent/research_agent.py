import logging
import sys
import kuzu

from langchain.prompts import PromptTemplate

from motleycrew.storage import MotleyKuzuGraphStore
from motleycrew.tool.llm_tool import LLMTool

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


QUESTION_PRIORITIZATION_TEMPLATE = PromptTemplate(
    template=(
        "You are provided with the following list of questions:"
        " {unanswered_questions} \n"
        " Your task is to choose one question from the above list"
        " that is the most pertinent to the following query:\n"
        " '{original_question}' \n"
        " Respond with one question out of the provided list of questions."
        " Return the questions as it is without any edits."
        " Format your response like:\n"
        " #. question"
    ),
    input_variables=["unanswered_questions", "original_question"],
)


class KnowledgeGainingOrchestrator:
    def __init__(self, db_path: str):
        self.db = kuzu.Database(db_path)
        self.storage = MotleyKuzuGraphStore(
            self.db, node_table_schema={"question": "STRING", "answer": "STRING", "context": "STRING"}
        )

        self.question_prioritization_tool = LLMTool(
            name="question_prioritization_tool",
            description="find the most important question",
            prompt=QUESTION_PRIORITIZATION_TEMPLATE,
        )
        self.question_generation_tool = None

    def get_unanswered_questions(self, only_without_children: bool = False) -> list[dict]:
        if only_without_children:
            query = "MATCH (n1:{}) WHERE n1.answer IS NULL AND NOT (n1)-[:{}]->(:{}) RETURN n1;".format(
                self.storage.node_table_name, self.storage.rel_table_name, self.storage.node_table_name
            )
        else:
            query = "MATCH (n1:{}) WHERE n1.answer IS NULL RETURN n1;".format(self.storage.node_table_name)

        query_result = self.storage.run_query(query)
        return [row[0] for row in query_result]  # flatten

    def __call__(self, query: str, max_iter: int):
        self.storage.create_entity({"question": query})

        for iter_n in range(max_iter):
            logging.info("====== Iteration %s of %s ======", iter_n, max_iter)

            unanswered_questions = self.get_unanswered_questions(only_without_children=True)
            logging.info("Loaded unanswered questions: %s", unanswered_questions)

            tool_input = "\n".join(f"{i}. {question}" for i, question in enumerate(unanswered_questions))
            most_pertinent_question_raw = self.question_prioritization_tool.invoke(tool_input)
            logging.info("Most pertinent question according to the tool: %s", most_pertinent_question_raw)

            i, most_pertinent_question_text = most_pertinent_question_raw.split(".", 1)
            assert i < len(unanswered_questions)

            most_pertinent_question = unanswered_questions[i]
            assert most_pertinent_question_text.strip() == most_pertinent_question["question"].strip()

            logging.info("Generating new questions")


if __name__ == "__main__":
    from pathlib import Path
    import shutil

    here = Path(__file__).parent
    db_path = here / "research_db"
    shutil.rmtree(db_path, ignore_errors=True)

    orchestrator = KnowledgeGainingOrchestrator(db_path=str(db_path))
