"""
Code derived from: https://github.com/run-llama/llama_index/blob/802064aee72b03ab38ead0cda780cfa3e37ce728/llama-index-integrations/graph_stores/llama-index-graph-stores-kuzu/llama_index/graph_stores/kuzu/base.py
Kùzu graph store index.
"""

from typing import Any, Dict, List, Optional

import kuzu


class MotleyQuestionGraphStore:
    IS_SUBQUESTION_PREDICATE = "IS_SUBQUESTION"

    def __init__(
        self,
        database: Any,
        node_table_name: str = "question",
        rel_table_name: str = "links",
        **kwargs: Any,
    ) -> None:
        self.database = database
        self.connection = kuzu.Connection(database)
        self.node_table_name = node_table_name
        self.rel_table_name = rel_table_name
        self.init_schema()

    def init_schema(self) -> None:
        """Initialize schema if the tables do not exist."""
        node_tables = self.connection._get_node_table_names()
        if self.node_table_name not in node_tables:
            self.connection.execute(
                "CREATE NODE TABLE %s (ID SERIAL, question STRING, answer STRING, context STRING[], PRIMARY KEY(ID))"
                % self.node_table_name
            )
        rel_tables = self.connection._get_rel_table_names()
        rel_tables = [rel_table["name"] for rel_table in rel_tables]
        if self.rel_table_name not in rel_tables:
            self.connection.execute(
                "CREATE REL TABLE {} (FROM {} TO {}, predicate STRING)".format(
                    self.rel_table_name, self.node_table_name, self.node_table_name
                )
            )

    @property
    def client(self) -> Any:
        return self.connection

    def check_question_exists(self, question_id: int) -> bool:
        is_exists_result = self.connection.execute(
            "MATCH (n:%s) WHERE n.ID = $question_id RETURN n.ID" % self.node_table_name,
            {"question_id": question_id},
        )
        return is_exists_result.has_next()

    def get_question(self, question_id: int) -> Optional[dict]:
        query = """
            MATCH (n1:%s)
            WHERE n1.ID = $question_id
            RETURN n1;
        """
        prepared_statement = self.connection.prepare(query % self.node_table_name)
        query_result = self.connection.execute(prepared_statement, {"question_id": question_id})

        if query_result.has_next():
            row = query_result.get_next()
            return row[0]

    def get_subquestions(self, question_id: int) -> List[int]:
        query = """
                    MATCH (n1:%s)-[r:%s]->(n2:%s)
                    WHERE n1.ID = $question_id
                    AND r.predicate = $is_subquestion_predicate
                    RETURN n2.ID;
                """
        prepared_statement = self.connection.prepare(
            query % (self.node_table_name, self.rel_table_name, self.node_table_name)
        )
        query_result = self.connection.execute(
            prepared_statement,
            {
                "question_id": question_id,
                "is_subquestion_predicate": MotleyQuestionGraphStore.IS_SUBQUESTION_PREDICATE,
            },
        )
        retval = []
        while query_result.has_next():
            row = query_result.get_next()
            retval.append(row[0])
        return retval

    def create_question(self, question: str) -> int:
        create_result = self.connection.execute(
            "CREATE (n:%s {question: $question}) " "RETURN n.ID" % self.node_table_name,
            {"question": question},
        )
        assert create_result.has_next()
        return create_result.get_next()[0]

    def create_subquestion(self, question_id: int, subquestion: str) -> int:
        def create_subquestion_rel(connection: Any, question_id: int, subquestion_id: int) -> None:
            connection.execute(
                (
                    "MATCH (n1:{}), (n2:{}) WHERE n1.ID = $question_id AND n2.ID = $subquestion_id "
                    "CREATE (n1)-[r:{} {{predicate: $is_subquestion_predicate}}]->(n2)"
                ).format(self.node_table_name, self.node_table_name, self.rel_table_name),
                {
                    "question_id": question_id,
                    "subquestion_id": subquestion_id,
                    "is_subquestion_predicate": MotleyQuestionGraphStore.IS_SUBQUESTION_PREDICATE,
                },
            )

        if not self.check_question_exists(question_id):
            raise Exception(f"No question with id {question_id}")

        subquestion_id = self.create_question(subquestion)
        create_subquestion_rel(self.connection, question_id=question_id, subquestion_id=subquestion_id)
        return subquestion_id

    def delete_question(self, question_id: int) -> None:
        """Deletes question and its relations."""

        def delete_rels(connection: Any, question_id: int) -> None:
            connection.execute(
                "MATCH (n1:{})-[r:{}]->(n2:{}) WHERE n1.ID = $question_id DELETE r".format(
                    self.node_table_name, self.rel_table_name, self.node_table_name
                ),
                {"question_id": question_id},
            )
            connection.execute(
                "MATCH (n1:{})<-[r:{}]-(n2:{}) WHERE n1.ID = $question_id DELETE r".format(
                    self.node_table_name, self.rel_table_name, self.node_table_name
                ),
                {"question_id": question_id},
            )

        def delete_question(connection: Any, question_id: int) -> None:
            connection.execute(
                "MATCH (n:%s) WHERE n.ID = $question_id DELETE n" % self.node_table_name,
                {"question_id": question_id},
            )

        delete_rels(self.connection, question_id)
        delete_question(self.connection, question_id)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        node_table_name: str = "entity",
        rel_table_name: str = "links",
    ) -> "MotleyQuestionGraphStore":
        """Load from persist dir."""
        try:
            import kuzu
        except ImportError:
            raise ImportError("Please install kuzu: pip install kuzu")
        database = kuzu.Database(persist_dir)
        return cls(database, node_table_name, rel_table_name)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MotleyQuestionGraphStore":
        """Initialize graph store from configuration dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Graph store.
        """
        return cls(**config_dict)


if __name__ == "__main__":
    from pathlib import Path

    here = Path(__file__).parent
    db_path = here / "test1"
    db = kuzu.Database(str(db_path))
    graph_store = MotleyQuestionGraphStore(db)

    q1_id = graph_store.create_question("q1")
    assert graph_store.get_question(q1_id)["question"] == "q1"

    q2_id = graph_store.create_subquestion(q1_id, "q2")
    q3_id = graph_store.create_subquestion(q1_id, "q3")
    q4_id = graph_store.create_subquestion(q3_id, "q4")

    assert set(graph_store.get_subquestions(q1_id)) == {q2_id, q3_id}
    assert set(graph_store.get_subquestions(q3_id)) == {q4_id}

    graph_store.delete_question(q4_id)
    assert graph_store.get_question(q4_id) is None
    assert not graph_store.get_subquestions(q3_id)

    print(f"docker run -p 8000:8000  -v {db_path}:/database --rm kuzudb/explorer: latest")
    print("MATCH (A)-[r]->(B) RETURN *;")
