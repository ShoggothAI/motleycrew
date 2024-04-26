from llama_index.core.graph_stores.types import GraphStore

from .question_answerer import AnswerSubQuestionTool


def answer_orchestrator(graph: GraphStore):
    last_question = None
    answerer = AnswerSubQuestionTool(graph=graph)
    while True:
        questions = graph.get_unanswered_available_questions()
        if not len(questions):
            return last_question
        else:
            last_question = questions[0]
            answerer.invoke({"question": last_question})
    return graph.retrieve(last_question.id)
