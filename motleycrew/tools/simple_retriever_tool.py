""" Module description """
import os.path

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from motleycrew.tools import MotleyTool
from motleycrew.applications.research_agent.question import Question


class SimpleRetrieverTool(MotleyTool):
    def __init__(self, DATA_DIR, PERSIST_DIR, return_strings_only: bool = False):
        """ Description

        Args:
            DATA_DIR (str):
            PERSIST_DIR (str):
            return_strings_only (:obj:`bool`, optional):
        """
        tool = make_retriever_langchain_tool(
            DATA_DIR, PERSIST_DIR, return_strings_only=return_strings_only
        )
        super().__init__(tool)


class RetrieverToolInput(BaseModel, arbitrary_types_allowed=True):
    """Input for the Retriever Tool.

    Attributes:
        question (Question):

    """

    question: Question = Field(
        description="The input question for which to retrieve relevant data."
    )


def make_retriever_langchain_tool(DATA_DIR, PERSIST_DIR, return_strings_only: bool = False):
    """ Description

    Args:
        DATA_DIR (str):
        PERSIST_DIR (str):
        return_strings_only (:obj:`bool`, optional):

    Returns:

    """
    text_embedding_model = "text-embedding-ada-002"
    embeddings = OpenAIEmbedding(model=text_embedding_model)

    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(
            documents, transformations=[SentenceSplitter(chunk_size=512), embeddings]
        )
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    retriever = index.as_retriever(
        similarity_top_k=10,
        embed_model=embeddings,
    )

    def call_retriever(question: Question) -> list:
        out = retriever.retrieve(question.question)
        if return_strings_only:
            return [node.text for node in out]
        return out

    retriever_tool = StructuredTool.from_function(
        func=call_retriever,
        name="information_retriever_tool",
        description="Useful for running a natural language query against a"
        " knowledge base and retrieving a set of relevant documents.",
        args_schema=RetrieverToolInput,
    )
    return retriever_tool


if __name__ == "__main__":

    # check if storage already exists
    here = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(here, "mahabharata/text/TinyTales")

    PERSIST_DIR = "../../examples/research_agent/storage"

    retriever_tool = SimpleRetrieverTool(DATA_DIR, PERSIST_DIR)
    response2 = retriever_tool.invoke(
        {"question": Question(question="What are the most interesting facts about Arjuna?")}
    )
    print(response2)
