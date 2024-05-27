import logging
from typing import List, Optional

from langchain_core.runnables import Runnable

from motleycrew.tasks import Task
from ...tasks.task_unit import TaskUnitType
from motleycrew.tools import MotleyTool
from motleycrew.crew import MotleyCrew
from .question import Question, QuestionGenerationTaskUnit
from .question_generator import QuestionGeneratorTool
from .question_prioritizer import QuestionPrioritizerTool


class QuestionTask(Task):
    def __init__(
        self,
        question: str,
        query_tool: MotleyTool,
        crew: MotleyCrew,
        max_iter: int = 10,
        name: str = "QuestionTask",
    ):
        # Need to supply the crew already at this stage
        # because need to use the graph store in constructor
        super().__init__(name=name, task_unit_class=QuestionGenerationTaskUnit, crew=crew)

        self.max_iter = max_iter
        self.n_iter = 0
        self.question = Question(question=question)
        self.graph_store.insert_node(self.question)
        self.question_prioritization_tool = QuestionPrioritizerTool()
        self.question_generation_tool = QuestionGeneratorTool(
            query_tool=query_tool, graph=self.graph_store
        )

    def get_next_unit(self) -> QuestionGenerationTaskUnit | None:
        if self.done:
            return None

        unanswered_questions = self.get_unanswered_questions(only_without_children=True)
        logging.info("Loaded unanswered questions: %s", unanswered_questions)

        if not len(unanswered_questions):
            return None

        most_pertinent_question = self.question_prioritization_tool.invoke(
            {
                "original_question": self.question,
                "unanswered_questions": unanswered_questions,
            }
        )
        logging.info("Most pertinent question according to the tool: %s", most_pertinent_question)
        return QuestionGenerationTaskUnit(question=most_pertinent_question)

    def register_completed_unit(self, unit: TaskUnitType) -> None:
        logging.info("==== Completed iteration %s of %s ====", self.n_iter + 1, self.max_iter)
        self.n_iter += 1
        if self.n_iter >= self.max_iter:
            self.set_done(True)

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        return self.question_generation_tool

    def get_unanswered_questions(self, only_without_children: bool = False) -> list[Question]:
        if only_without_children:
            query = (
                "MATCH (n1:{}) WHERE n1.answer IS NULL AND NOT (n1)-[]->(:{}) RETURN n1;".format(
                    Question.get_label(), Question.get_label()
                )
            )
        else:
            query = "MATCH (n1:{}) WHERE n1.answer IS NULL RETURN n1;".format(Question.get_label())

        query_result = self.graph_store.run_cypher_query(query, container=Question)
        return query_result
