from typing import Optional
import tempfile
import os

from motleycrew.common import Defaults
from motleycrew.common import GraphStoreType
from motleycrew.common import logger
from motleycrew.storage import MotleyKuzuGraphStore, MotleyGraphStore


def init_graph_store(
    graph_store_type: str = Defaults.DEFAULT_GRAPH_STORE_TYPE,
    db_path: Optional[str] = None,
) -> MotleyGraphStore:
    """Create and initialize a graph store with the given parameters.

    Args:
        graph_store_type: Type of the graph store to use.
        db_path: Path to the database for the graph store.

    Returns:
        Initialized graph store.
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
