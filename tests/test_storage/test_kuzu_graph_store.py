import pytest

from typing import Optional
import random
import kuzu
from pydantic import BaseModel
from motleycrew.storage import MotleyKuzuGraphStore


class Entity(BaseModel):
    int_param: int
    optional_str_param: Optional[str] = None
    optional_list_str_param: Optional[list[str]] = None


@pytest.fixture
def database(tmpdir):
    db_path = tmpdir / "test_db"
    db = kuzu.Database(str(db_path))
    return db


class TestMotleyKuzuGraphStore:
    def test_create_new_entity(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        created_entity = graph_store.create_entity(entity)
        assert getattr(created_entity, "_id", None) is not None

    def test_create_entity_and_retrieve(self, database):
        graph_store = MotleyKuzuGraphStore(database)

        entity = Entity(int_param=1, optional_str_param="test", optional_list_str_param=["a", "b"])
        created_entity = graph_store.create_entity(entity)
        created_entity_id = getattr(created_entity, "_id", None)
        assert created_entity_id is not None

        retrieved_entity = graph_store.get_entity_by_class_and_id(
            entity_class=Entity, entity_id=created_entity_id
        )
        assert created_entity == retrieved_entity

    def test_create_entity_with_id_already_set(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        setattr(entity, "_id", 1)
        with pytest.raises(AssertionError):
            graph_store.create_entity(entity)

    def test_check_entity_exists_true(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)

        graph_store.create_entity(entity)
        assert graph_store.check_entity_exists(entity) is True

    def test_check_entity_exists_false(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        assert graph_store.check_entity_exists(entity) is False

        setattr(entity, "_id", 1)
        assert graph_store.check_entity_exists(entity) is False

        graph_store._ensure_entity_table(entity)
        assert graph_store.check_entity_exists(entity) is False

    def test_create_relation(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)
        graph_store.create_entity(entity1)
        graph_store.create_entity(entity2)

        graph_store.create_relation(from_entity=entity1, to_entity=entity2, predicate="p")

        assert graph_store.check_relation_exists(from_entity=entity1, to_entity=entity2)
        assert graph_store.check_relation_exists(
            from_entity=entity1, to_entity=entity2, predicate="p"
        )
        assert not graph_store.check_relation_exists(
            from_entity=entity1, to_entity=entity2, predicate="q"
        )
        assert not graph_store.check_relation_exists(from_entity=entity2, to_entity=entity1)

    def test_upsert_triplet(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)
        graph_store.upsert_triplet(from_entity=entity1, to_entity=entity2, predicate="p")

        assert graph_store.check_entity_exists(entity1)
        assert graph_store.check_entity_exists(entity2)

        assert graph_store.check_relation_exists(from_entity=entity1, to_entity=entity2)
        assert graph_store.check_relation_exists(
            from_entity=entity1, to_entity=entity2, predicate="p"
        )
        assert not graph_store.check_relation_exists(
            from_entity=entity1, to_entity=entity2, predicate="q"
        )
        assert not graph_store.check_relation_exists(from_entity=entity2, to_entity=entity1)

    def test_entities_do_not_exist(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        assert not graph_store.check_entity_exists(entity1)
        assert not graph_store.check_entity_exists(entity2)

        assert not graph_store.check_relation_exists(from_entity=entity1, to_entity=entity2)
        assert not graph_store.check_relation_exists(
            from_entity=entity2, to_entity=entity1, predicate="p"
        )

    def test_relation_does_not_exist(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        assert not graph_store.check_relation_exists(from_entity=entity1, to_entity=entity2)
        assert not graph_store.check_relation_exists(from_entity=entity2, to_entity=entity1)

        graph_store.create_entity(entity1)
        graph_store.create_entity(entity2)

        assert not graph_store.check_relation_exists(from_entity=entity1, to_entity=entity2)
        assert not graph_store.check_relation_exists(from_entity=entity2, to_entity=entity1)

    def test_delete_entity(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        graph_store.create_entity(entity)
        assert graph_store.check_entity_exists(entity) is True

        graph_store.delete_entity(entity)
        assert graph_store.check_entity_exists(entity) is False

    def test_delete_entity_with_relations(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        graph_store.create_entity(entity1)
        graph_store.create_entity(entity2)
        graph_store.create_relation(from_entity=entity1, to_entity=entity2, predicate="p")
        assert graph_store.check_relation_exists(from_entity=entity1, to_entity=entity2) is True

        graph_store.delete_entity(entity1)
        assert graph_store.check_entity_exists(entity1) is False
        assert graph_store.check_entity_exists(entity2) is True
        assert graph_store.check_relation_exists(from_entity=entity1, to_entity=entity2) is False

    def test_set_property(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity = Entity(int_param=1)
        graph_store.create_entity(entity)
        assert entity.optional_str_param is None
        assert graph_store.get_entity_by_class_and_id(Entity, entity._id).optional_str_param is None

        graph_store.set_property(entity, "optional_str_param", "test")
        assert entity.optional_str_param == "test"
        assert (
            graph_store.get_entity_by_class_and_id(Entity, entity._id).optional_str_param == "test"
        )

        graph_store.set_property(entity, "optional_list_str_param", ["a", "b"])
        assert entity.optional_list_str_param == ["a", "b"]
        assert graph_store.get_entity_by_class_and_id(
            Entity, entity._id
        ).optional_list_str_param == ["a", "b"]

    def test_run_cypher_query(self, database):
        graph_store = MotleyKuzuGraphStore(database)
        entity1 = Entity(int_param=1)
        entity2 = Entity(int_param=2)

        graph_store.create_entity(entity1)
        graph_store.create_entity(entity2)
        graph_store.create_relation(from_entity=entity1, to_entity=entity2, predicate="p")

        query = """
        MATCH (a:entity {int_param: 1})-[r]->(b:entity {int_param: 2})
        RETURN a, r, b
        """
        result = graph_store.run_cypher_query(query)
        assert len(result) == 1
        assert len(result[0]) == 3

        a, r, b = result[0]
        assert a["int_param"] == 1
        assert b["int_param"] == 2
        assert r["predicate"] == "p"
