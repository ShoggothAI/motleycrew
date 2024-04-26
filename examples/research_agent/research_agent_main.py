import os.path


from examples.research_agent.retriever_tool import make_retriever_tool

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
