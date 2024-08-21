from typing import Optional

import pytest

from motleycrew.common import GraphStoreType
from motleycrew.storage import MotleyGraphNode
from motleycrew.storage import MotleyKuzuGraphStore
from tests.test_storage import GraphStoreFixtures


class Entity(MotleyGraphNode):
    int_param: int
    optional_str_param: Optional[str] = None
    optional_list_str_param: Optional[list[str]] = None


class TestMotleyGraphStore(GraphStoreFixtures):
    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_insert_new_node(self, graph_store):
        entity = Entity(int_param=1)
        created_entity = graph_store.insert_node(entity)
        assert created_entity.id is not None
        assert entity.id is not None  # mutated in place

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_insert_node_and_retrieve(self, graph_store):
        entity = Entity(int_param=1, optional_str_param="test", optional_list_str_param=["a", "b"])
        inserted_entity = graph_store.insert_node(entity)
        assert inserted_entity.id is not None

        retrieved_entity = graph_store.get_node_by_class_and_id(
            node_class=Entity, node_id=inserted_entity.id
        )
        assert inserted_entity == retrieved_entity

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_check_node_exists_true(self, graph_store):
        entity = Entity(int_param=1)

        graph_store.insert_node(entity)
        assert graph_store.check_node_exists(entity)

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_check_node_exists_false(self, graph_store):
        entity = Entity(int_param=1)
        assert graph_store.check_node_exists(entity) is False

        MotleyKuzuGraphStore._set_node_id(node=entity, node_id=2)
        assert graph_store.check_node_exists(entity) is False

        graph_store.ensure_node_table(Entity)
        assert graph_store.check_node_exists(entity) is False

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_create_relation(self, graph_store):
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)
        graph_store.insert_node(entity1)
        graph_store.insert_node(entity2)

        graph_store.create_relation(from_node=entity1, to_node=entity2, label="p")

        assert graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert graph_store.check_relation_exists(from_node=entity1, to_node=entity2, label="p")
        assert not graph_store.check_relation_exists(from_node=entity1, to_node=entity2, label="q")
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1)

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_upsert_triplet(self, graph_store):
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)
        graph_store.upsert_triplet(from_node=entity1, to_node=entity2, label="p")

        assert graph_store.check_node_exists(entity1)
        assert graph_store.check_node_exists(entity2)

        assert graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert graph_store.check_relation_exists(from_node=entity1, to_node=entity2, label="p")
        assert not graph_store.check_relation_exists(from_node=entity1, to_node=entity2, label="q")
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1)

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_nodes_do_not_exist(self, graph_store):
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        assert not graph_store.check_node_exists(entity1)
        assert not graph_store.check_node_exists(entity2)

        assert not graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1, label="p")

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_relation_does_not_exist(self, graph_store):
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        assert not graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1)

        graph_store.insert_node(entity1)
        graph_store.insert_node(entity2)

        assert not graph_store.check_relation_exists(from_node=entity1, to_node=entity2)
        assert not graph_store.check_relation_exists(from_node=entity2, to_node=entity1)

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_delete_node(self, graph_store):
        entity = Entity(int_param=1)
        graph_store.insert_node(entity)
        assert graph_store.check_node_exists(entity) is True

        graph_store.delete_node(entity)
        assert graph_store.check_node_exists(entity) is False

        entity.int_param = 2  # check that entity is not frozen

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_delete_entity_with_relations(self, graph_store):
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        graph_store.insert_node(entity1)
        graph_store.insert_node(entity2)
        graph_store.create_relation(from_node=entity1, to_node=entity2, label="p")
        assert graph_store.check_relation_exists(from_node=entity1, to_node=entity2) is True

        graph_store.delete_node(entity1)
        assert graph_store.check_node_exists(entity1) is False
        assert graph_store.check_node_exists(entity2) is True
        assert graph_store.check_relation_exists(from_node=entity1, to_node=entity2) is False

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_set_property(self, graph_store):
        entity = Entity(int_param=1)
        graph_store.insert_node(entity)
        assert entity.optional_str_param is None
        assert graph_store.get_node_by_class_and_id(Entity, entity.id).optional_str_param is None

        entity.optional_str_param = "test"
        assert graph_store.get_node_by_class_and_id(Entity, entity.id).optional_str_param == "test"

        entity.optional_list_str_param = ["a", "b"]
        assert graph_store.get_node_by_class_and_id(Entity, entity.id).optional_list_str_param == [
            "a",
            "b",
        ]

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_run_cypher_query(self, graph_store):
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

    @pytest.mark.parametrize("graph_store", GraphStoreType.ALL, indirect=True)
    def test_run_cypher_query_with_container(self, graph_store):
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
