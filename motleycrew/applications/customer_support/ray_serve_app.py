from pathlib import Path

import ray
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from ray import serve

from motleycrew.applications.customer_support.support_agent import (
    CustomerChatTool,
    IssueTreeViewTool,
    ResolveIssueTool,
    SupportAgent,
    SupportAgentContext,
)
from motleycrew.common import configure_logging, logger
from motleycrew.common.llms import LLMFramework, init_llm
from motleycrew.storage import MotleyKuzuGraphStore

app = FastAPI()


class WebSocketCommunicationInterface:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send_message_to_customer(self, message: str) -> str:
        await self.websocket.send_json({"type": "agent_message", "content": message})
        response = await self.websocket.receive_text()
        return response

    async def escalate_to_human_agent(self) -> None:
        await self.websocket.send_json(
            {"type": "escalation", "content": "Your issue has been escalated to a human agent."}
        )

    async def resolve_issue(self, resolution: str) -> str:
        await self.websocket.send_json({"type": "resolution", "content": resolution})


@serve.deployment(num_replicas=3, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@serve.ingress(app)
class SupportAgentDeployment:
    def __init__(self):
        configure_logging(verbose=True)
        load_dotenv()

        database = MotleyKuzuGraphStore.from_persist_dir(
            str(Path(__file__).parent / "issue_tree_db"), read_only=True
        )
        self.graph_store = database
        self.llm = init_llm(LLMFramework.LANGCHAIN)

    @app.get("/")
    async def root(self):
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.websocket("/ws")
    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json(
            {"type": "agent_message", "content": "Hello! How can I help you?"}
        )

        communication_interface = WebSocketCommunicationInterface(websocket)

        context = SupportAgentContext(self.graph_store, communication_interface)
        issue_tree_view_tool = IssueTreeViewTool(context)
        customer_chat_tool = CustomerChatTool(context)
        resolve_issue_tool = ResolveIssueTool(context)

        agent = SupportAgent(
            issue_tree_view_tool=issue_tree_view_tool,
            customer_chat_tool=customer_chat_tool,
            resolve_issue_tool=resolve_issue_tool,
            llm=self.llm,
        )

        try:
            while True:
                data = await websocket.receive_text()
                resolution = await agent.ainvoke({"prompt": data})
                if resolution.additional_kwargs.get("escalate"):
                    await communication_interface.escalate_to_human_agent()
                else:
                    await communication_interface.resolve_issue(resolution.content)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")


def main():
    configure_logging(verbose=True)

    deployment = SupportAgentDeployment.bind()

    ray.init(address="auto")
    serve.run(deployment)


if __name__ == "__main__":
    main()
