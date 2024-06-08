import pytest

from typing import Optional
import kuzu
from motleycrew.storage import MotleyGraphNode
from motleycrew.storage import MotleyKuzuGraphStore


class Entity(MotleyGraphNode):
    int_param: int
    optional_str_param: Optional[str] = None
    optional_list_str_param: Optional[list[str]] = None


@pytest.fixture
def database(tmpdir):
    db_path = tmpdir / "test_db"
    db = kuzu.Database(str(db_path))
    return db


class TestMotleyKuzuGraphStore:
    def test_set_get_node_id(self):
        entity = Entity(int_param=1)
        MotleyKuzuGraphStore._set_node_id(node=entity, node_id=2)
        assert entity.id == 2

    def test_insert_new_node(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        created_entity = graph_store.insert_node(entity)
        assert created_entity.id is not None
        assert entity.id is not None  # mutated in place

    def test_insert_node_and_retrieve(self, database):
        graph_store = MotleyKuzuGraphStore(database)

        entity = Entity(
            int_param=1, optional_str_param="test", optional_list_str_param=["a", "b"]
        )
        inserted_entity = graph_store.insert_node(entity)
        assert inserted_entity.id is not None

        retrieved_entity = graph_store.get_node_by_class_and_id(
            node_class=Entity, node_id=inserted_entity.id
        )
        assert inserted_entity == retrieved_entity

    def test_insert_node_with_id_already_set(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        MotleyKuzuGraphStore._set_node_id(node=entity, node_id=2)
        with pytest.raises(AssertionError):
            graph_store.insert_node(entity)

    def test_check_node_exists_true(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)

        graph_store.insert_node(entity)
        assert graph_store.check_node_exists(entity)

    def test_check_node_exists_false(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        assert graph_store.check_node_exists(entity) is False

        MotleyKuzuGraphStore._set_node_id(node=entity, node_id=2)
        assert graph_store.check_node_exists(entity) is False

        graph_store.ensure_node_table(Entity)
        assert graph_store.check_node_exists(entity) is False

    def test_create_relation(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)
        graph_store.insert_node(entity1)
        graph_store.insert_node(entity2)

        graph_store.create_relation(from_node=entity1, to_node=entity2, label="p")

        assert graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert graph_store.check_relation_exists(
            from_node=entity1, to_node=entity2, label="p"
        )
        assert not graph_store.check_relation_exists(
            from_node=entity1, to_node=entity2, label="q"
        )
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1)

    def test_upsert_triplet(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)
        graph_store.upsert_triplet(from_node=entity1, to_node=entity2, label="p")

        assert graph_store.check_node_exists(entity1)
        assert graph_store.check_node_exists(entity2)

        assert graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert graph_store.check_relation_exists(
            from_node=entity1, to_node=entity2, label="p"
        )
        assert not graph_store.check_relation_exists(
            from_node=entity1, to_node=entity2, label="q"
        )
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1)

    def test_nodes_do_not_exist(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        assert not graph_store.check_node_exists(entity1)
        assert not graph_store.check_node_exists(entity2)

        assert not graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert not graph_store.check_relation_exists(
            from_node=entity2, to_node=entity1, label="p"
        )

    def test_relation_does_not_exist(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        assert not graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1)

        graph_store.insert_node(entity1)
        graph_store.insert_node(entity2)

        assert not graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1)

    def test_delete_node(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        graph_store.insert_node(entity)
        assert graph_store.check_node_exists(entity) is True

        graph_store.delete_node(entity)
        assert graph_store.check_node_exists(entity) is False

        entity.int_param = 2  # check that entity is not frozen

    def test_delete_entity_with_relations(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        graph_store.insert_node(entity1)
        graph_store.insert_node(entity2)
        graph_store.create_relation(from_node=entity1, to_node=entity2, label="p")
        assert (
            graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
            is True
        )

        graph_store.delete_node(entity1)
        assert graph_store.check_node_exists(entity1) is False
        assert graph_store.check_node_exists(entity2) is True
        assert (
            graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
            is False
        )

    def test_set_property(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        graph_store.insert_node(entity)
        assert entity.optional_str_param is None
        assert (
            graph_store.get_node_by_class_and_id(Entity, entity.id).optional_str_param
            is None
        )

        entity.optional_str_param = "test"
        assert (
            graph_store.get_node_by_class_and_id(Entity, entity.id).optional_str_param
            == "test"
        )

        entity.optional_list_str_param = ["a", "b"]
        assert graph_store.get_node_by_class_and_id(
            Entity, entity.id
        ).optional_list_str_param == [
            "a",
            "b",
        ]

    def test_run_cypher_query(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        graph_store.insert_node(entity1)
        graph_store.insert_node(entity2)
        graph_store.create_relation(from_node=entity1, to_node=entity2, label="p")

        query = """
        MATCH (a:Entity {int_param: 1})-[r]->(b:Entity {int_param: 2})
        RETURN a, r, b
        """
        result = graph_store.run_cypher_query(query)
        assert len(result) == 1
        assert len(result[0]) == 3

        a, r, b = result[0]
        assert a["int_param"] == 1
        assert b["int_param"] == 2
        assert r["_label"] == "p"

    def test_run_cypher_query_with_container(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2, optional_list_str_param=["a", "b"])

        graph_store.insert_node(entity1)
        graph_store.insert_node(entity2)

        query = """
        MATCH (a:Entity)
        WHERE a.int_param = 2
        RETURN a
        """
        result = graph_store.run_cypher_query(query, container=Entity)
        assert len(result) == 1
        assert isinstance(result[0], Entity)
        assert result[0].int_param == 2
        assert result[0].optional_list_str_param == ["a", "b"]
