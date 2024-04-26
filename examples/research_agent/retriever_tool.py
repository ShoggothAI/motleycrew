import os.path

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from llama_index.core.tools import RetrieverTool


from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from motleycrew.tool import MotleyTool


def make_retriever_tool(DATA_DIR, PERSIST_DIR):
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

    class RetrieverToolInput(BaseModel):
        """Input for the Retriever Tool."""

        question: str = Field(
            description="The input question for which to retrieve relevant data."
        )

    def call_retriever(question: str) -> str:
        out = retriever.retrieve(question)
        return out

    retriever_tool = StructuredTool.from_function(
        func=call_retriever,
        name="Information retriever tool",
        description="Useful for running a natural language query against a"
        " knowledge base and retrieving a set of relevant documents.",
        args_schema=RetrieverToolInput,
    )
    return MotleyTool.from_langchain_tool(retriever_tool)


if __name__ == "__main__":

    # check if storage already exists
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.realpath(os.path.join(here, "../../.."))
    DATA_DIR = os.path.join(root, "mahabharata/text/TinyTales")

    PERSIST_DIR = "./storage"

    retriever_tool = make_retriever_tool(DATA_DIR, PERSIST_DIR)
    response2 = retriever_tool.invoke(
        {"question": "What are the most interesting facts about Arjuna?"}
    )
    print("yay!")
