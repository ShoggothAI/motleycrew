from typing import List, Optional

from langchain_core.runnables import Runnable

from motleycrew.applications.research_agent.question import Question
from motleycrew.applications.research_agent.question_answerer import AnswerSubQuestionTool
from motleycrew.common import logger
from motleycrew.crew import MotleyCrew
from motleycrew.tasks import Task, TaskUnit
from motleycrew.tools import MotleyTool


class QuestionAnsweringTaskUnit(TaskUnit):
    question: Question


class AnswerTask(Task):
    """Task to answer a question based on the notes and sub-questions."""

    def __init__(
        self,
        crew: MotleyCrew,
        answer_length: int = 1000,
    ):
        super().__init__(
            name="AnswerTask",
            task_unit_class=QuestionAnsweringTaskUnit,
            crew=crew,
            allow_async_units=True,
        )
        self.answer_length = answer_length
        self.answerer = AnswerSubQuestionTool(
            graph=self.graph_store, answer_length=self.answer_length
        )

    def get_next_unit(self) -> QuestionAnsweringTaskUnit | None:
        """Choose an unanswered question to answer.

        The question should have a context and no unanswered subquestions."""

        query = (
            "MATCH (n1:{}) "
            "WHERE n1.answer IS NULL AND n1.context IS NOT NULL "
            "AND NOT EXISTS {{MATCH (n1)-[]->(n2:{}) "
            "WHERE n2.answer IS NULL AND n2.context IS NOT NULL}} "
            "RETURN n1"
        ).format(Question.get_label(), Question.get_label())

        query_result = self.graph_store.run_cypher_query(query, container=Question)
        logger.info("Available questions: %s", query_result)

        existing_units = self.get_units()
        for question in query_result:
            if not any(unit.question.question == question.question for unit in existing_units):
                return QuestionAnsweringTaskUnit(question=question)

        return None

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        return self.answerer
