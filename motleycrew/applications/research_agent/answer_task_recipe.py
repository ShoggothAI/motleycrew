import logging
from typing import List, Optional
from langchain_core.runnables import Runnable

from motleycrew.crew import MotleyCrew
from motleycrew.tool import MotleyTool
from motleycrew.tasks import TaskRecipe
from motleycrew.tasks.task import TaskType
from motleycrew.tasks import Task
from motleycrew.applications.research_agent.question import Question, QuestionAnsweringTask
from motleycrew.applications.research_agent.question_answerer import AnswerSubQuestionTool
from motleycrew.storage import MotleyGraphStore


class AnswerTaskRecipe(TaskRecipe):
    def __init__(
        self,
        crew: MotleyCrew,
        answer_length: int = 1000,
    ):
        super().__init__("AnswerTaskRecipe", crew)
        self.answer_length = answer_length
        self.answerer = AnswerSubQuestionTool(
            graph=self.graph_store, answer_length=self.answer_length
        )

    def identify_candidates(self) -> list[QuestionAnsweringTask]:
        query = (
            "MATCH (n1:{}) "
            "WHERE n1.answer IS NULL AND n1.context IS NOT NULL "
            "AND NOT EXISTS {{MATCH (n1)-[]->(n2:{}) "
            "WHERE n2.answer IS NULL AND n2.context IS NOT NULL}} "
            "RETURN n1"
        ).format(Question.get_label(), Question.get_label())

        query_result = self.graph_store.run_cypher_query(query, container=Question)
        logging.info("Available questions: %s", query_result)
        return [QuestionAnsweringTask(question=q) for q in query_result]

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        return self.answerer
