import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from motleycrew.agents.langchain import LangchainMotleyAgent
from motleycrew.applications.customer_support.communication import (
    CommunicationInterface,
    DummyCommunicationInterface,
)
from motleycrew.applications.customer_support.issue_tree import IssueData, IssueNode
from motleycrew.common import LLMFramework
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.common.llms import init_llm
from motleycrew.storage import MotleyKuzuGraphStore
from motleycrew.tools import MotleyTool


@dataclass
class SupportAgentContext:
    graph_store: MotleyKuzuGraphStore
    communication_interface: CommunicationInterface
    viewed_issues: List[IssueData] = field(default_factory=list)


class IssueTreeViewInput(BaseModel):
    categories_path: List[str] = Field(
        ..., description="Path as a list of consecutive subcategories of the issue tree."
    )


class IssueTreeViewTool(MotleyTool):
    def __init__(self, context: SupportAgentContext):
        super().__init__(
            name="IssueTreeViewTool",
            description="Tool that can view the issue tree at a given path.",
            args_schema=IssueTreeViewInput,
        )
        self.context = context

    def run(self, categories_path: List[str]):
        """
        Args:
            categories_path: list of consecutive subcategories of the issue tree.

        Returns:
            str: Pretty view of the issue tree at the given path.
                The view should include the subcategories, if any, and the issues at the leaf nodes.
        """
        current_node = self._get_root_node()

        for category in categories_path:
            current_node = self._get_child_node(current_node, category)
            if current_node is None:
                return f"Category not found: {category}"

        return self._format_node_view(current_node)

    def _get_root_node(self) -> Optional[IssueNode]:
        root_query = "MATCH (n:IssueNode) WHERE n.name = 'Root' RETURN n"
        roots = self.context.graph_store.run_cypher_query(root_query, container=IssueNode)
        return roots[0] if roots else None

    def _get_child_node(self, parent: IssueNode, category: str) -> Optional[IssueNode]:
        child_query = """
        MATCH (parent:IssueNode)-[:HAS_SUBCATEGORY]->(child:IssueNode)
        WHERE parent.id = $parent_id AND child.name = $category
        RETURN child
        """
        children = self.context.graph_store.run_cypher_query(
            child_query,
            parameters={"parent_id": parent.id, "category": category},
            container=IssueNode,
        )
        return children[0] if children else None

    def _format_node_view(self, node: IssueNode) -> str:
        result = f"Category: {node.name}\n"
        result += f"Description: {node.description}\n\n"

        subcategories = self._get_subcategories(node)
        if subcategories:
            result += "Subcategories:\n"
            for subcat in subcategories:
                result += f"- {subcat.name}\n"

        issues = self._get_issues(node)
        if issues:
            self.context.viewed_issues.extend(issues)

            result += "\nIssues:\n"
            for issue in issues:
                result += f"- {issue.description}\n"
                result += f"  Resolution: {issue.resolution}\n\n"

        return result

    def _get_subcategories(self, node: IssueNode) -> List[IssueNode]:
        subcat_query = """
        MATCH (parent:IssueNode)-[:HAS_SUBCATEGORY]->(child:IssueNode)
        WHERE parent.id = $parent_id
        RETURN child
        """
        return self.context.graph_store.run_cypher_query(
            subcat_query, parameters={"parent_id": node.id}, container=IssueNode
        )

    def _get_issues(self, node: IssueNode) -> List[IssueData]:
        issues_query = """
        MATCH (category:IssueNode)-[:HAS_ISSUE]->(issue:IssueData)
        WHERE category.id = $category_id
        RETURN issue
        """
        return self.context.graph_store.run_cypher_query(
            issues_query, parameters={"category_id": node.id}, container=IssueData
        )


class CustomerChatInput(BaseModel):
    message: str = Field(..., description="Message to send to the customer.")


class CustomerChatTool(MotleyTool):
    def __init__(self, context: SupportAgentContext):
        super().__init__(
            name="CustomerChatTool",
            description="Tool that can send a message to the customer and receive a response. "
            "Use it if you need to inquire additional details from the customer or to propose a solution.",
            args_schema=CustomerChatInput,
        )
        self.context = context

    async def arun(self, message: str) -> str:
        """
        Args:
            message: Message to send to the customer.

        Returns:
            str: The response from the customer.
        """
        return await self.context.communication_interface.send_message_to_customer(message)


class ResolveIssueInput(BaseModel):
    resolution: Optional[str] = Field(None, description="Resolution to the issue.")
    escalate: bool = Field(..., description="Whether to escalate the issue to a human agent.")


class ResolveIssueTool(MotleyTool):
    def __init__(self, context: SupportAgentContext):
        super().__init__(
            name="ResolveIssueTool",
            description="Tool that can resolve the issue or escalate it to a human agent.",
            args_schema=ResolveIssueInput,
            return_direct=True,
        )
        self.context = context

    def run(self, resolution: Optional[str] = None, escalate: bool = False) -> AIMessage:
        """
        Args:
            resolution: Resolution to the issue.
            escalate: Whether to escalate the issue to a human agent.

        Returns:
            AIMessage: The resolution to the issue with `escalate` value in the additional kwargs.
        """
        if escalate:
            content = f"I am escalating this issue to a human agent. Resolution: {resolution}"
            return AIMessage(content=content, additional_kwargs={"escalate": True})
        else:
            if not resolution:
                raise InvalidOutput("Resolution must be provided when not escalating.")
            if not self.context.viewed_issues:
                raise InvalidOutput("You must view at least some past issues before resolving.")
            return AIMessage(content=resolution, additional_kwargs={"escalate": False})


SUPPORT_AGENT_PROMPT = """You are a customer support agent. Your goal is to answer customer's questions.
You must only rely on the knowledge of past issues and the current dialogue with the customer.
Categorized past issues and their resolutions are stored in the issue tree.
The leaf nodes of the tree are the individual issues and the internal nodes are their categories.

Here are the top-level nodes (categories) of the issue tree:
{issue_tree_top_level_categories}

You should start by determining the category of the issue.
If you cannot determine the category, you should ask the customer for additional information, until you have a clear understanding of the issue.
You should not directly ask the customer which category the issue belongs to, but rather ask specific questions to narrow down the issue.
Don't mention categories in your questions to the customer.
After determining the category, you should view the issue tree at the given path.
Call the IssueTreeViewTool, providing the path to the category as a list of consecutive subcategories of the issue tree,
starting with the top-level category.
Repeat this process until you reach the list of issues at the leaf nodes of the tree.

Carefully review the list of issues in the category and their resolutions.
If you can answer the customer's question based on the past issues and their resolutions, do so.
In this case, you must call ResolveIssueTool with a clear and direct resolution to the issue.
The resolution must contain direct instructions on how to resolve the issue, or a constructive answer to the customer's question.

If you think you made a mistake in categorizing the issue, you can use the IssueTreeViewTool to inspect another category.
If you are sure that you cannot answer the question based on known information, you must escalate the issue to a human agent by calling ResolveIssueTool with `escalate` set to `true`.
If you need to ask the customer for additional information, you should use the CustomerChatTool.

Communicate with the customer only through the provided tools. Never return responses in plain text.
"""


REMINDER = """Remember to only rely on the knowledge of past issues and the current dialogue with the customer.
Never make up an answer. If you do not know the answer, you must escalate the issue to a human agent.

Proceed with calling the relevant tool:"""


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SUPPORT_AGENT_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        MessagesPlaceholder(variable_name="input"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="additional_notes", optional=True),
        ("system", REMINDER),
    ]
)


def format_chat_history(chat_history: list) -> str:
    formatted_history = "Summary of previous conversation:\n"
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_history += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"Agent: {message.content}\n"
    return [SystemMessage(content=formatted_history)]


def create_support_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    issue_tree_view_tool: IssueTreeViewTool,
) -> Runnable:
    tools_for_llm = list(tools)
    llm_with_tools = llm.bind_tools(tools=tools_for_llm)

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"]),
            additional_notes=lambda x: x.get("additional_notes") or [],
            chat_history=lambda x: (
                format_chat_history(x.get("chat_history")) if x.get("chat_history") else []
            ),
            issue_tree_top_level_categories=lambda x: issue_tree_view_tool.invoke(
                {"categories_path": []}
            ),
        )
        | PROMPT_TEMPLATE
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    return agent


class SupportAgent(LangchainMotleyAgent):
    def __init__(
        self,
        issue_tree_view_tool: IssueTreeViewTool,
        customer_chat_tool: CustomerChatTool,
        resolve_issue_tool: ResolveIssueTool,
        llm: Optional[BaseChatModel] = None,
    ):
        if llm is None:
            llm = init_llm(LLMFramework.LANGCHAIN)

        def agent_factory(tools: dict[str, MotleyTool]) -> AgentExecutor:
            tools_for_langchain = [t.to_langchain_tool() for t in tools.values()]

            return AgentExecutor(
                agent=create_support_agent(llm, tools_for_langchain, issue_tree_view_tool),
                tools=tools_for_langchain,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                max_iterations=10,
            )

        super().__init__(
            name="SupportAgent",
            description="Agent that can answer customer support questions",
            tools=[issue_tree_view_tool, customer_chat_tool, resolve_issue_tool],
            agent_factory=agent_factory,
            force_output_handler=True,
            input_as_messages=True,
            chat_history=True,
        )


async def main():
    load_dotenv()

    graph_store = MotleyKuzuGraphStore.from_persist_dir(
        str(Path(__file__).parent / "issue_tree_db")
    )
    communication_interface = DummyCommunicationInterface()

    context = SupportAgentContext(graph_store, communication_interface)

    issue_tree_view_tool = IssueTreeViewTool(context)
    customer_chat_tool = CustomerChatTool(context)
    resolve_issue_tool = ResolveIssueTool(context)

    agent = SupportAgent(
        issue_tree_view_tool=issue_tree_view_tool,
        customer_chat_tool=customer_chat_tool,
        resolve_issue_tool=resolve_issue_tool,
        llm=init_llm(LLMFramework.LANGCHAIN),
    )

    print("Starting Customer Support Agent Demo")

    customer_query = "I forgot my password."
    print(f"\nCustomer: {customer_query}")

    response = await agent.ainvoke(
        {
            "prompt": customer_query,
        }
    )
    print(f"Agent: {response.content}")

    followup_query = "What if I forgot my email?"
    print(f"\nCustomer: {followup_query}")

    response = await agent.ainvoke(
        {
            "prompt": followup_query,
        }
    )
    print(f"Agent: {response.content}")

    print("\nDemo completed.")


if __name__ == "__main__":
    asyncio.run(main())
