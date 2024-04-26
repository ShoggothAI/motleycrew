import os.path

import pandas as pd

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

text_embedding_model = "text-embedding-ada-002"
embeddings = OpenAIEmbedding(model=text_embedding_model)

# check if storage already exists
PERSIST_DIR = "./storage"
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.realpath(os.path.join(here, "../../.."))
DATA_DIR = os.path.join(root, "mahabharata/text/TinyTales")
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

# Either way we can now query the index
query_engine = index.as_query_engine(
    similarity_top_k=10, embeddings=embeddings, response_mode="tree_summarize"
)
response = query_engine.query(
    "What are the most interesting facts about Arjuna?",
)
print(response)
