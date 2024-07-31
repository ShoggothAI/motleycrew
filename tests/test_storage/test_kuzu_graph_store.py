from typing import Optional

import kuzu
import pytest

from motleycrew.common import GraphStoreType
from motleycrew.storage import MotleyGraphNode
from motleycrew.storage import MotleyKuzuGraphStore
from tests.test_storage import GraphStoreFixtures


class Entity(MotleyGraphNode):
    int_param: int
    optional_str_param: Optional[str] = None
    optional_list_str_param: Optional[list[str]] = None


@pytest.fixture
def database(tmpdir):
    db_path = tmpdir / "test_db"
    db = kuzu.Database(str(db_path))
    return db


class TestMotleyKuzuGraphStore(GraphStoreFixtures):
    def test_set_get_node_id(self):
        entity = Entity(int_param=1)
        MotleyKuzuGraphStore._set_node_id(node=entity, node_id=2)
        assert entity.id == 2

    @pytest.mark.parametrize("graph_store", [GraphStoreType.KUZU], indirect=True)
    def test_insert_node_with_id_already_set(self, graph_store):
        entity = Entity(int_param=1)
        MotleyKuzuGraphStore._set_node_id(node=entity, node_id=2)
        with pytest.raises(AssertionError):
            graph_store.insert_node(entity)
