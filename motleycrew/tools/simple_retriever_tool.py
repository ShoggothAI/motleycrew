import os
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from motleycrew.applications.research_agent.question import Question
from motleycrew.tools import MotleyTool


class SimpleRetrieverTool(MotleyTool):
    """A simple retriever tool that retrieves relevant documents from a local knowledge base."""

    def __init__(
        self,
        data_dir: str,
        persist_dir: str,
        return_strings_only: bool = False,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        """
        Args:
            data_dir: Path to the directory containing the documents.
            persist_dir: Path to the directory to store the index.
            return_strings_only: Whether to return only the text of the retrieved documents.
        """
        tool = make_retriever_langchain_tool(
            data_dir, persist_dir, return_strings_only=return_strings_only
        )
        super().__init__(
            tool=tool, return_direct=return_direct, exceptions_to_reflect=exceptions_to_reflect
        )


class RetrieverToolInput(BaseModel, arbitrary_types_allowed=True):
    """Input for the retriever tool."""

    question: Question = Field(
        description="The input question for which to retrieve relevant data."
    )


def make_retriever_langchain_tool(data_dir, persist_dir, return_strings_only: bool = False):
    text_embedding_model = "text-embedding-ada-002"
    embeddings = OpenAIEmbedding(model=text_embedding_model)

    if not os.path.exists(persist_dir):
        # load the documents and create the index
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(
            documents, transformations=[SentenceSplitter(chunk_size=512), embeddings]
        )
        # store it for later
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
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
