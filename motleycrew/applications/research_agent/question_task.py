""" Module description """

from typing import List, Optional

from langchain_core.runnables import Runnable

from motleycrew.tasks import Task
from motleycrew.tasks.task_unit import TaskUnitType
from motleycrew.tools import MotleyTool
from motleycrew.crew import MotleyCrew
from motleycrew.common import TaskUnitStatus
from .question import Question, QuestionGenerationTaskUnit
from .question_generator import QuestionGeneratorTool
from .question_prioritizer import QuestionPrioritizerTool
from motleycrew.common import logger


class QuestionTask(Task):
    def __init__(
        self,
        question: str,
        query_tool: MotleyTool,
        crew: MotleyCrew,
        max_iter: int = 10,
        allow_async_units: bool = False,
        name: str = "QuestionTask",
    ):
        """Description

        Args:
            question (str):
            query_tool (MotleyTool):
            crew (MotleyCrew):
            max_iter (:obj:`int`, optional):
            name (:obj:`str`, optional):
        """
        # Need to supply the crew already at this stage
        # because need to use the graph store in constructor
        super().__init__(
            name=name,
            task_unit_class=QuestionGenerationTaskUnit,
            crew=crew,
            allow_async_units=allow_async_units,
        )

        self.max_iter = max_iter
        self.n_iter = 0
        self.question = Question(question=question)
        self.graph_store.insert_node(self.question)
        self.question_prioritization_tool = QuestionPrioritizerTool()
        self.question_generation_tool = QuestionGeneratorTool(
            query_tool=query_tool, graph=self.graph_store
        )

    def get_next_unit(self) -> QuestionGenerationTaskUnit | None:
        """Description

        Returns:
            QuestionGenerationTaskUnit
        """
        if self.done or self.n_iter >= self.max_iter:
            return None

        unanswered_questions = self.get_unanswered_questions(only_without_children=True)
        logger.info("Loaded unanswered questions: %s", unanswered_questions)

        existing_units = self.get_units()
        question_candidates = []
        for question in unanswered_questions:
            if not any(unit.question.question == question.question for unit in existing_units):
                question_candidates.append(question)

        if not len(question_candidates):
            return None

        most_pertinent_question = self.question_prioritization_tool.invoke(
            {
                "original_question": self.question,
                "unanswered_questions": question_candidates,
            }
        )
        logger.info("Most pertinent question according to the tool: %s", most_pertinent_question)
        return QuestionGenerationTaskUnit(question=most_pertinent_question)

    def register_started_unit(self, unit: TaskUnitType) -> None:
        """Description

        Args:
            unit (TaskUnitType):

        Returns:

        """
        logger.info("==== Started iteration %s of %s ====", self.n_iter + 1, self.max_iter)
        self.n_iter += 1

    def register_completed_unit(self, unit: TaskUnitType) -> None:
        if self.n_iter >= self.max_iter:
            self.set_done(True)

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        """Description

        Args:
            tools (List[MotleyTool]):

        Returns:
            Runnable
        """
        return self.question_generation_tool

    def get_unanswered_questions(self, only_without_children: bool = False) -> list[Question]:
        """Description

        Args:
            only_without_children (:obj:`bool`, optional):

        Returns:
            list[Question]
        """
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
