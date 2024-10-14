from typing import List, Optional
from dotenv import load_dotenv
import shutil
from pathlib import Path

import pandas
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from motleycrew.common.llms import LLMFramework, init_llm
from motleycrew.storage import MotleyGraphNode
from motleycrew.storage.kuzu_graph_store import MotleyKuzuGraphStore


class IssueNode(MotleyGraphNode):
    name: str
    description: Optional[str] = None


class Issue(BaseModel):
    title: str
    description: str
    resolution: str


class IssueData(MotleyGraphNode):
    description: str
    resolution: str


def insert_issue_data(graph_store: MotleyKuzuGraphStore, issue: Issue, categories_path: List[str]):
    # Ensure the IssueNode and IssueData tables exist
    graph_store.ensure_node_table(IssueNode)
    graph_store.ensure_node_table(IssueData)
    graph_store.ensure_relation_table(IssueNode, IssueNode, "HAS_SUBCATEGORY")
    graph_store.ensure_relation_table(IssueNode, IssueData, "HAS_ISSUE")

    # Start from the root, create it if it doesn't exist
    root_query = "MATCH (n:IssueNode) WHERE n.name = 'Root' RETURN n"
    root_nodes = graph_store.run_cypher_query(root_query, container=IssueNode)

    if not root_nodes:
        root_node = IssueNode(name="Root", description="Root of the issue tree")
        current_node = graph_store.insert_node(root_node)
    else:
        current_node = root_nodes[0]

    # Traverse or create the category path
    for category in categories_path:
        child_query = """
        MATCH (parent:IssueNode)-[:HAS_SUBCATEGORY]->(child:IssueNode)
        WHERE parent.id = $parent_id AND child.name = $category
        RETURN child
        """
        children = graph_store.run_cypher_query(
            child_query,
            parameters={"parent_id": current_node.id, "category": category},
            container=IssueNode,
        )

        if children:
            current_node = children[0]
        else:
            new_node = IssueNode(name=category, description=f"Category: {category}")
            new_node = graph_store.insert_node(new_node)
            graph_store.create_relation(current_node, new_node, "HAS_SUBCATEGORY")
            current_node = new_node

    # Create and link the issue data
    issue_data = IssueData(description=issue.description, resolution=issue.resolution)
    issue_data = graph_store.insert_node(issue_data)
    graph_store.create_relation(current_node, issue_data, "HAS_ISSUE")

    return issue_data


class IssueCategorizer:
    def __init__(self, graph_store: MotleyKuzuGraphStore, max_depth: int = 3):
        self.graph_store = graph_store
        self.max_depth = max_depth
        self.llm = init_llm(LLMFramework.LANGCHAIN)

    def _get_or_create_root(self) -> IssueNode:
        self.graph_store.ensure_node_table(IssueNode)

        root_query = """
        MATCH (n:IssueNode)
        WHERE n.name = 'Root'
        RETURN n
        """
        results = self.graph_store.run_cypher_query(root_query, container=IssueNode)

        if results:
            return results[0]
        else:
            root = IssueNode(name="Root", description="Root of the issue tree")
            return self.graph_store.insert_node(root)

    def _get_children(self, parent: IssueNode) -> List[IssueNode]:
        self.graph_store.ensure_node_table(IssueNode)
        self.graph_store.ensure_relation_table(IssueNode, IssueNode, "HAS_SUBCATEGORY")

        children_query = """
        MATCH (parent:IssueNode)-[:HAS_SUBCATEGORY]->(child:IssueNode)
        WHERE parent.id = $parent_id
        RETURN child
        """
        return self.graph_store.run_cypher_query(
            children_query, parameters={"parent_id": parent.id}, container=IssueNode
        )

    def _create_child(self, parent: IssueNode, name: str, description: str) -> IssueNode:
        child = IssueNode(name=name, description=description)
        child = self.graph_store.insert_node(child)
        self.graph_store.create_relation(parent, child, "HAS_SUBCATEGORY")
        return child

    def _categorize(self, issue: Issue, categories: List[IssueNode]) -> Optional[str]:
        prompt = ChatPromptTemplate.from_template(
            """
        Given the following issue and list of categories, determine the most appropriate category.
        If none of the categories fit well, respond with "NEW" and suggest a name for a new category.

        Issue:
        Description: {description}
        Resolution: {resolution}

        Categories:
        {categories}

        Response format:
        Category: <category_name or "NEW">
        New Category (if applicable): <suggested_new_category_name>
        Reason: <brief explanation>

        Response:
        """
        )

        categories_str = "\n".join([f"- {cat.name}: {cat.description}" for cat in categories])
        response = self.llm.invoke(
            prompt.format(
                description=issue.description,
                resolution=issue.resolution,
                categories=categories_str,
            )
        )

        lines = response.content.strip().split("\n")
        category = lines[0].split(": ")[1].strip()

        if category == "NEW":
            return lines[1].split(": ")[1].strip()
        return category

    def categorize_issue(self, issue: Issue, insert_issue: bool = False):
        current_node = self._get_or_create_root()
        depth = 0

        while depth < self.max_depth:
            children = self._get_children(current_node)

            if not children:
                # If there are no children, create a new category
                new_category_name = self._categorize(issue, [])
                current_node = self._create_child(current_node, new_category_name, "")
            else:
                category = self._categorize(issue, children)

                if category == "NEW":
                    new_category_name = self._categorize(issue, [])
                    current_node = self._create_child(current_node, new_category_name, "")
                else:
                    matching_child = next(
                        (child for child in children if child.name == category), None
                    )
                    if matching_child:
                        current_node = matching_child
                    else:
                        current_node = self._create_child(current_node, category, "")

            depth += 1

        # Create and store the IssueData node
        issue_data = IssueData(
            title=issue.title, description=issue.description, resolution=issue.resolution
        )
        if insert_issue:
            issue_node = self.graph_store.insert_node(issue_data)

            # Link the IssueData node to the leaf category node
            self.graph_store.create_relation(current_node, issue_node, "HAS_ISSUE")
        else:
            issue_node = None

        return current_node, issue_node


def main():
    load_dotenv()

    # (Re-)initialize the graph store
    db_path = Path(__file__).parent / "issue_tree_db"
    shutil.rmtree(db_path, ignore_errors=True)
    graph_store = MotleyKuzuGraphStore.from_persist_dir(db_path)

    issues = pandas.read_csv(Path(__file__).parent / "example_issues.csv")

    for _, row in issues.iterrows():
        # Create IssueData node
        issue_data = IssueData(description=row["issue_description"], resolution=row["solution"])

        # Split the category string into a list
        categories = row["category"].split(" > ")

        # Create or find the category path
        leaf_node = insert_issue_data(graph_store, issue_data, categories)
        print(f"Processed issue: {row['issue_id']} - Category: {row['category']}")

    # Create the categorizer
    categorizer = IssueCategorizer(graph_store, max_depth=3)

    # Example usage
    issue = Issue(
        title="Can't log in to my account",
        description="I've been trying to log in to my account for the past hour, but it keeps saying my password is incorrect. I'm sure I'm using the right password.",
        resolution="Guided the customer through the password reset process. The issue was resolved after the customer set a new password.",
    )

    print(f"Example issue: {issue}")
    leaf_node, issue_node = categorizer.categorize_issue(issue)
    print(f"Issue categorized as: {leaf_node.name}")


if __name__ == "__main__":
    main()
