from typing import Optional

from motleycrew.storage.graph_node import MotleyGraphNode

REPR_CONTEXT_LENGTH_LIMIT = 30


class Question(MotleyGraphNode):
    """Represents a question node in the graph."""

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
