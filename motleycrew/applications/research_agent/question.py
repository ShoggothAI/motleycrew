from typing import Optional
from dataclasses import dataclass
import json

from motleycrew.storage.graph_node import MotleyGraphNode
from motleycrew.tasks import Task

REPR_CONTEXT_LENGTH_LIMIT = 30


class Question(MotleyGraphNode):
    question: str
    answer: Optional[str] = None
    context: Optional[list[str]] = None

    def __repr__(self):
        if self.context and len(self.context):
            context_repr = '", "'.join(self.context)
            if len(context_repr) > REPR_CONTEXT_LENGTH_LIMIT:
                context_repr = '["' + context_repr[:REPR_CONTEXT_LENGTH_LIMIT] + "...]"
            else:
                context_repr = '["' + context_repr + '"]'
        else:
            context_repr = str(self.context)

        return "Question(id={}, question={}, answer={}, context={})".format(
            self.id, self.question, self.answer, context_repr
        )


class QuestionGenerationTask(Task):
    question: Question


class QuestionAnsweringTask(Task):
    question: Question
