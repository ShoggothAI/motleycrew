"""
Code derived from: https://github.com/run-llama/llama_index/blob/802064aee72b03ab38ead0cda780cfa3e37ce728/llama-index-integrations/graph_stores/llama-index-graph-stores-kuzu/llama_index/graph_stores/kuzu/base.py
Kùzu graph store index.
"""

from typing import Any, Dict, List, Optional

import json
import kuzu


class MotleyKuzuGraphStore:
    def __init__(
        self,
        database: Any,
        node_table_schema: dict[str, str],
        node_table_name: str = "entity",
        rel_table_name: str = "links",
        **kwargs: Any,
    ) -> None:
        self.database = database
        self.connection = kuzu.Connection(database)
        self.node_table_schema = node_table_schema
        self.node_table_name = node_table_name
        self.rel_table_name = rel_table_name
        self.init_schema()

    def init_schema(self) -> None:
        """Initialize schema if the tables do not exist."""
        node_tables = self.connection._get_node_table_names()
        if self.node_table_name not in node_tables:
            node_table_schema_expr = ", ".join(
                ["id SERIAL"]
                + [f"{name} {datatype}" for name, datatype in self.node_table_schema.items()]
                + ["PRIMARY KEY(id)"]
            )
            self.connection.execute("CREATE NODE TABLE {} ({})".format(self.node_table_name, node_table_schema_expr))

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

    def check_entity_exists(self, entity_id: int) -> bool:
        is_exists_result = self.connection.execute(
            "MATCH (n:%s) WHERE n.id = $entity_id RETURN n.id" % self.node_table_name,
            {"entity_id": entity_id},
        )
        return is_exists_result.has_next()

    def get_entity(self, entity_id: int) -> Optional[dict]:
        query = """
            MATCH (n1:%s)
            WHERE n1.id = $entity_id
            RETURN n1;
        """
        prepared_statement = self.connection.prepare(query % self.node_table_name)
        query_result = self.connection.execute(prepared_statement, {"entity_id": entity_id})

        if query_result.has_next():
            row = query_result.get_next()
            item = row[0]
            return item

    def create_entity(self, entity: dict) -> int:
        """Create a new entity and return its id"""
        create_result = self.connection.execute(
            "CREATE (n:{} $entity) RETURN n.id".format(self.node_table_name),
            {"entity": entity},
        )
        assert create_result.has_next()
        return create_result.get_next()[0]

    def create_rel(self, from_id: int, to_id: int, predicate: str) -> None:
        self.connection.execute(
            (
                "MATCH (n1:{}), (n2:{}) WHERE n1.id = $from_id AND n2.id = $to_id "
                "CREATE (n1)-[r:{} {{predicate: $predicate}}]->(n2)"
            ).format(self.node_table_name, self.node_table_name, self.rel_table_name),
            {
                "from_id": from_id,
                "to_id": to_id,
                "predicate": predicate,
            },
        )

    def delete_entity(self, entity_id: int) -> None:
        """Delete a given entity and its relations"""

        def delete_rels(connection: Any, entity_id: int) -> None:
            # Undirected relation removal is not supported for some reason
            connection.execute(
                "MATCH (n1:{})-[r:{}]->(n2:{}) WHERE n1.id = $entity_id DELETE r;"
                "MATCH (n1:{})<-[r:{}]-(n2:{}) WHERE n1.id = $entity_id DELETE r".format(
                    self.node_table_name,
                    self.rel_table_name,
                    self.node_table_name,
                    self.node_table_name,
                    self.rel_table_name,
                    self.node_table_name,
                ),
                {"entity_id": entity_id},
            )
            connection.execute(
                "MATCH (n1:{})<-[r:{}]-(n2:{}) WHERE n1.id = $entity_id DELETE r".format(
                    self.node_table_name, self.rel_table_name, self.node_table_name
                ),
                {"entity_id": entity_id},
            )

        def delete_entity(connection: Any, entity_id: int) -> None:
            connection.execute(
                "MATCH (n:%s) WHERE n.id = $entity_id DELETE n" % self.node_table_name,
                {"entity_id": entity_id},
            )

        delete_rels(self.connection, entity_id)
        delete_entity(self.connection, entity_id)

    def set_property(self, entity_id: int, property_name: str, property_value: Any):
        query = """
                    MATCH (n1:{})
                    WHERE n1.id = $entity_id
                    SET n1.{} = $property_value;
                """
        prepared_statement = self.connection.prepare(query.format(self.node_table_name, property_name))
        self.connection.execute(prepared_statement, {"entity_id": entity_id, "property_value": property_value})

    def run_query(self, query: str, parameters: Optional[dict] = None) -> list[list]:
        """Run a Cypher query and return the results"""
        query_result = self.connection.execute(query=query, parameters=parameters)
        retval = []
        while query_result.has_next():
            retval.append(query_result.get_next())
        return retval

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        node_table_schema: dict[str, str],
        node_table_name: str = "entity",
        rel_table_name: str = "links",
    ) -> "MotleyKuzuGraphStore":
        """Load from persist dir."""
        try:
            import kuzu
        except ImportError:
            raise ImportError("Please install kuzu: pip install kuzu")
        database = kuzu.Database(persist_dir)
        return cls(database, node_table_schema, node_table_name, rel_table_name)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MotleyKuzuGraphStore":
        """Initialize graph store from configuration dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Graph store.
        """
        return cls(**config_dict)


if __name__ == "__main__":
    from pathlib import Path
    import shutil

    here = Path(__file__).parent
    db_path = here / "test1"
    shutil.rmtree(db_path, ignore_errors=True)
    db = kuzu.Database(str(db_path))
    graph_store = MotleyKuzuGraphStore(
        db, node_table_schema={"question": "STRING", "answer": "STRING", "context": "STRING"}
    )

    IS_SUBQUESTION_PREDICATE = "is_subquestion"

    q1_id = graph_store.create_entity({"question": "q1"})
    assert graph_store.get_entity(q1_id)["question"] == "q1"

    q2_id = graph_store.create_entity({"question": "q2"})
    q3_id = graph_store.create_entity({"question": "q3"})
    q4_id = graph_store.create_entity({"question": "q4"})
    graph_store.create_rel(q1_id, q2_id, IS_SUBQUESTION_PREDICATE)
    graph_store.create_rel(q1_id, q3_id, IS_SUBQUESTION_PREDICATE)
    graph_store.create_rel(q3_id, q4_id, IS_SUBQUESTION_PREDICATE)

    graph_store.delete_entity(q4_id)
    assert graph_store.get_entity(q4_id) is None

    graph_store.set_property(q2_id, property_name="answer", property_value="a2")
    graph_store.set_property(q3_id, property_name="", property_value=["c3_1", "c3_2"])

    assert graph_store.get_entity(q2_id)["answer"] == "a2"
    assert graph_store.get_entity(q3_id)["context"] == ["c3_1", "c3_2"]

    print(f"docker run -p 8000:8000  -v {db_path}:/database --rm kuzudb/explorer: latest")
    print("MATCH (A)-[r]->(B) RETURN *;")
