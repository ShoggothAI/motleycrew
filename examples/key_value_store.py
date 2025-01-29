from typing import List

from motleycrew import MotleyCrew
from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
from motleycrew.storage.kv_store_domain import SimpleRetrievableObject
from motleycrew.tasks import SimpleTask
from motleycrew.tools import MotleyTool


class ObjectInsertionTool(MotleyTool):
    def run(self, query: str) -> List[str]:
        response1 = SimpleRetrievableObject(
            id="321",
            name="test",
            description="test",
            payload={"test": 2.35, "blah": 3.14},
        )
        response2 = SimpleRetrievableObject(
            id="aaa",
            name="test1",
            description="another dummy object",
            payload={"test": 2.35, "blah": 3.14},
        )
        self.agent.kv_store[response1.id] = response1
        self.agent.kv_store[response2.id] = response2
        return [response1.summary, response2.summary]


class ObjectFetcherTool(MotleyTool):
    def run(self, object_id: str) -> str:
        result = self.agent.kv_store[object_id]
        print(result.payload)
        return "success!"


if __name__ == "__main__":
    from motleycrew.common.logging import configure_logging

    configure_logging()

    instructions = """Your job is to first call the Object Insertion Tool and get back from it the metadata
    of some data objects; each metadata will include an id.
    You should then call the Object Fetcher Tool with the id of one of the objects you got back from the Object Insertion Tool.
    """

    my_agent = ReActToolCallingMotleyAgent(
        tools=[
            ObjectInsertionTool(
                name="Object_Insertion_tool",
                description="When called with a query, returns back the "
                "metadata of one or several relevant cached objects ",
            ),
            ObjectFetcherTool(
                name="Object_Fetcher_tool",
                description="When called with an object id, prints the payload of the object",
            ),
        ],
        description="Object retrieval agent",
        name="Object retrieval agent",
        verbose=True,
    )
    crew = MotleyCrew()
    task = SimpleTask(crew=crew, agent=my_agent, description=instructions)
    crew.run()

    print("yay!")
