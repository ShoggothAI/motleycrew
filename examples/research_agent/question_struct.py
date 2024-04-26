from typing import Optional
from dataclasses import dataclass
import json


@dataclass
class Question:
    id: Optional[int] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    context: Optional[list[str]] = None

    def serialize(self):
        data = {}

        if self.id:
            data["id"] = json.dumps(self.id)
        if self.context:
            data["question"] = json.dumps(self.question)
        if self.context:
            data["answer"] = json.dumps(self.answer)
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
