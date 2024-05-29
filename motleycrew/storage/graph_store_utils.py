""" Module description """
import tempfile
from typing import Optional
import os

from motleycrew.common import Defaults
from motleycrew.common import GraphStoreType
from motleycrew.common import logger
from motleycrew.storage import MotleyKuzuGraphStore


def init_graph_store(
    graph_store_type: str = Defaults.DEFAULT_GRAPH_STORE_TYPE,
    db_path: Optional[str] = None,
):
    """ Description

    Args:
        graph_store_type (:obj:`str`, optional):
        db_path (:obj:`str`, optional):

    Returns:

    """
    if graph_store_type == GraphStoreType.KUZU:
        import kuzu

        if db_path is None:
            logger.info("No db_path provided, creating temporary directory for database")
            db_path = os.path.join(tempfile.mkdtemp(), "kuzu_db")

        logger.info("Using Kuzu graph store with path: %s", db_path)
        db = kuzu.Database(db_path)
        return MotleyKuzuGraphStore(db)

    raise ValueError(f"Unknown graph store type: {graph_store_type}")
