from typing import Optional
from dataclasses import dataclass
import json

REPR_CONTEXT_LENGTH_LIMIT = 30


@dataclass
class Question:
    id: Optional[int] = None
    question: Optional[str] = None
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

    def serialize(self):
        data = {}

        if self.id:
            data["id"] = self.id
        if self.question:
            data["question"] = self.question
        if self.answer:
            data["answer"] = self.answer
        if self.context:
            data["context"] = json.dumps(self.context)

        return data

    @staticmethod
    def deserialize(data: dict):
        context_raw = data["context"]
        if context_raw:
            context = json.loads(context_raw)
        else:
            context = None

        return Question(id=data["id"], question=data["question"], answer=data["answer"], context=context)
