import uuid
import os

from dotenv import load_dotenv


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import PGVector

load_dotenv("../.env")

DB_PASSWORD = os.environ["SUPABASE_PASSWORD"]
DB_DBUSER = os.environ["SUPABASE_DBUSER"]
DB_DATABASE = os.environ["SUPABASE_DATABASE"]
DB_HOST = os.environ["SUPABASE_HOST"]
DB_PORT = os.environ["SUPABASE_PORT"]
DB_CONN_STRING = (
    f"postgresql://{DB_DBUSER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
)

text_embedding_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=text_embedding_model)

mahabharata_store = PGVector(
    collection_name="mahabharata",
    connection_string=DB_CONN_STRING,
    embedding_function=embeddings,
)
