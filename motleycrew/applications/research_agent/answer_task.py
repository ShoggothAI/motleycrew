""" Module description"""
from typing import List, Optional
from langchain_core.runnables import Runnable

from motleycrew.crew import MotleyCrew
from motleycrew.tools import MotleyTool
from motleycrew.tasks import Task
from motleycrew.tasks.task_unit import TaskUnitType
from motleycrew.tasks import TaskUnit
from motleycrew.applications.research_agent.question import Question, QuestionAnsweringTaskUnit
from motleycrew.applications.research_agent.question_answerer import AnswerSubQuestionTool
from motleycrew.storage import MotleyGraphStore
from motleycrew.common import logger


class AnswerTask(Task):
    def __init__(
        self,
        crew: MotleyCrew,
        answer_length: int = 1000,
    ):
        """ Description

        Args:
            crew (MotleyCrew):
            answer_length (:obj:`int`, optional):
        """
        super().__init__(name="AnswerTask", task_unit_class=QuestionAnsweringTaskUnit, crew=crew)
        self.answer_length = answer_length
        self.answerer = AnswerSubQuestionTool(
            graph=self.graph_store, answer_length=self.answer_length
        )

    def get_next_unit(self) -> QuestionAnsweringTaskUnit | None:
        """ Description

        Returns:
            QuestionAnsweringTaskUnit | None:
        """
        query = (
            "MATCH (n1:{}) "
            "WHERE n1.answer IS NULL AND n1.context IS NOT NULL "
            "AND NOT EXISTS {{MATCH (n1)-[]->(n2:{}) "
            "WHERE n2.answer IS NULL AND n2.context IS NOT NULL}} "
            "RETURN n1"
        ).format(Question.get_label(), Question.get_label())

        query_result = self.graph_store.run_cypher_query(query, container=Question)
        logger.info("Available questions: %s", query_result)
        if not query_result:
            return None
        else:
            return QuestionAnsweringTaskUnit(question=query_result[0])

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        """ Description

        Args:
            tools (List[MotleyTool]):

        Returns:
            Runnable:
        """
        return self.answerer
