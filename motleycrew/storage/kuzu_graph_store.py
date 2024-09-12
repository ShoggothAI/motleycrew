"""
Code derived from: https://github.com/run-llama/llama_index/blob/802064aee72b03ab38ead0cda780cfa3e37ce728/llama-index-integrations/graph_stores/llama-index-graph-stores-kuzu/llama_index/graph_stores/kuzu/base.py

Kùzu graph store index.
"""

import json
import os
from typing import Any, Dict, Optional, Type, Collection

from kuzu import Connection, PreparedStatement, QueryResult

from motleycrew.common import logger
from motleycrew.storage import MotleyGraphNode
from motleycrew.storage import MotleyGraphNodeType
from motleycrew.storage import MotleyGraphStore


class MotleyKuzuGraphStore(MotleyGraphStore):
    """Kuzu graph store implementation for motleycrew."""

    ID_ATTR = "_id"

    JSON_CONTENT_PREFIX = "JSON__"

    PYTHON_TO_CYPHER_TYPES_MAPPING = {
        int: "INT64",  # TODO: enforce size when creating and updating nodes and relations
        Optional[int]: "INT64",
        str: "STRING",
        Optional[str]: "STRING",
        float: "DOUBLE",
        Optional[float]: "DOUBLE",
        bool: "BOOLEAN",
        Optional[bool]: "BOOLEAN",
    }

    def __init__(self, database: Any) -> None:
        """Initialize Kuzu graph store.

        Args:
            database: Kuzu database client
        """
        self.database = database
        self.connection = Connection(database)

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.database_path})"

    def __str__(self):
        return self.__repr__()

    @property
    def database_path(self) -> str:
        return os.path.abspath(self.database.database_path)

    def _execute_query(
        self, query: str | PreparedStatement, parameters: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute a query, logging it for debugging purposes.

        Args:
            query: Cypher query or prepared statement.
            parameters: Query parameters.

        Returns:
            Query result.
        """
        logger.debug("Executing query: %s", query)
        if parameters:
            logger.debug("with parameters: %s", parameters)

        # TODO: retries?
        return self.connection.execute(query=query, parameters=parameters)

    def _check_node_table_exists(self, label: str) -> bool:
        """Check if a table for storing nodes with given label exists in the database.

        Args:
            label: Node label.

        Returns:
            Whether the table exists.
        """
        return label in self.connection._get_node_table_names()

    def _check_rel_table_exists(
        self,
        from_label: Optional[str] = None,
        to_label: Optional[str] = None,
        rel_label: Optional[str] = None,
    ) -> bool:
        """Check if a table for storing relations between nodes with given labels
        exists in the database.

        Args:
            from_label: Label of the source node.
            to_label: Label of the destination node.
            rel_label: Label of the relation.

        Returns:
            Whether the table exists.
        """
        for row in self.connection._get_rel_table_names():
            if (
                (rel_label is None or row["name"] == rel_label)
                and (from_label is None or row["src"] == from_label)
                and (to_label is None or row["dst"] == to_label)
            ):
                return True
        return False

    def _get_node_property_names(self, label: str) -> Collection[str]:
        """Get the names of properties for nodes with given label.

        Args:
            label: Node label.

        Returns:
            Collection of property names.
        """
        return self.connection._get_node_property_names(table_name=label)

    def ensure_node_table(self, node_class: Type[MotleyGraphNode]) -> str:
        """Create a table for storing nodes of that class if such does not already exist.
        If it does exist, create all missing columns.

        Args:
            node_class: Node Python class.

        Returns:
            Node table name.
        """
        table_name = node_class.get_label()
        if not self._check_node_table_exists(table_name):
            logger.info("Node table %s does not exist in the database, creating", table_name)
            self._execute_query(
                "CREATE NODE TABLE {} (id SERIAL, PRIMARY KEY(id))".format(table_name)
            )

        # Create missing property columns
        existing_property_names = self._get_node_property_names(node_class.get_label())
        for field_name, field in node_class.model_fields.items():
            if field_name not in existing_property_names:
                logger.info(
                    "Property %s not present in table for label %s, creating",
                    field_name,
                    node_class.get_label(),
                )
                cypher_type, is_json = (
                    MotleyKuzuGraphStore._get_cypher_type_and_is_json_by_python_type_annotation(
                        field.annotation
                    )
                )

                self._execute_query(
                    "ALTER TABLE {} ADD {} {}".format(table_name, field_name, cypher_type)
                )
        return table_name

    def ensure_relation_table(
        self, from_class: Type[MotleyGraphNode], to_class: Type[MotleyGraphNode], label: str
    ) -> None:
        """Create a table for storing relations from from_node-like nodes to to_node-like nodes,
        if such does not already exist.

        Args:
            from_class: Source node Python class.
            to_class: Destination node Python class.
            label: Relation label.
        """
        if not self._check_rel_table_exists(
            from_label=from_class.get_label(), to_label=to_class.get_label(), rel_label=label
        ):
            logger.info(
                "Relation table %s from %s to %s does not exist in the database, creating",
                label,
                from_class.get_label(),
                to_class.get_label(),
            )

            self._execute_query(
                "CREATE REL TABLE {} (FROM {} TO {})".format(
                    label, from_class.get_label(), to_class.get_label()
                )
            )

    def check_node_exists_by_class_and_id(
        self, node_class: Type[MotleyGraphNode], node_id: int
    ) -> bool:
        """Check if a node of given class with given id is present in the database.

        Args:
            node_class: Node Python class.
            node_id: Node id.

        Returns:
            Whether the node exists in the database.
        """
        if not self._check_node_table_exists(node_class.get_label()):
            return False

        is_exists_result = self._execute_query(
            "MATCH (n:{}) WHERE n.id = $node_id RETURN n.id".format(node_class.get_label()),
            {"node_id": node_id},
        )
        return is_exists_result.has_next()

    def check_node_exists(self, node: MotleyGraphNode) -> bool:
        """Check if the given node is present in the database.

        Args:
            node: Node to check.

        Returns:
            Whether the node exists in the database.
        """
        if node.id is None:
            return False  # for cases when id attribute is not set => node does not exist

        return self.check_node_exists_by_class_and_id(node_class=node.__class__, node_id=node.id)

    def check_relation_exists(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: Optional[str] = None
    ) -> bool:
        """Check if a relation exists between two nodes with given label.

        Args:
            from_node: Source node.
            to_node: Destination node.
            label: Relation label. If None, any relation is taken into account.

        Returns:
            Whether the relation exists in the database.
        """
        if from_node.id is None or to_node.id is None:
            return False

        if (
            not self._check_node_table_exists(from_node.get_label())
            or not self._check_node_table_exists(to_node.get_label())
            or not self._check_rel_table_exists(
                from_label=from_node.get_label(), to_label=to_node.get_label(), rel_label=label
            )
        ):
            return False

        query = (
            "MATCH (n1:{})-[r{}]->(n2:{}) "
            "WHERE n1.id = $from_node_id AND n2.id = $to_node_id "
            "RETURN r".format(
                from_node.get_label(),
                (":" + label) if label else "",
                to_node.get_label(),
            )
        )
        parameters = {
            "from_node_id": from_node.id,
            "to_node_id": to_node.id,
        }

        is_exists_result = self._execute_query(query=query, parameters=parameters)
        return is_exists_result.has_next()

    def get_node_by_class_and_id(
        self, node_class: Type[MotleyGraphNodeType], node_id: int
    ) -> Optional[MotleyGraphNodeType]:
        """Retrieve the node of given class with given id if it is present in the database.

        Args:
            node_class: Node Python class.
            node_id: Node id.

        Returns:
            Node object or None if it does not exist.
        """
        if not self._check_node_table_exists(node_class.get_label()):
            return None

        query = """
                    MATCH (n:{})
                    WHERE n.id = $node_id
                    RETURN n;
                """.format(
            node_class.get_label()
        )
        query_result = self._execute_query(query, {"node_id": node_id})

        if query_result.has_next():
            row = query_result.get_next()
            return self._deserialize_node(node_dict=row[0], node_class=node_class)

    def insert_node(self, node: MotleyGraphNodeType) -> MotleyGraphNodeType:
        """Insert a new node and populate its id.

        If the node table or some columns do not exist, this method also creates them.

        Args:
            node: Node to insert.

        Returns:
            Inserted node.
        """
        assert node.id is None, "Entity has its id set, looks like it is already in the DB"

        self.ensure_node_table(type(node))
        logger.info("Inserting new node with label %s: %s", node.get_label(), node)

        cypher_mapping, parameters = MotleyKuzuGraphStore._node_to_cypher_mapping_with_parameters(
            node
        )
        create_result = self._execute_query(
            "CREATE (n:{} {}) RETURN n".format(node.get_label(), cypher_mapping),
            parameters=parameters,
        )
        assert create_result.has_next()
        logger.info("Node created OK")

        created_object = create_result.get_next()[0]
        created_object_id = created_object.get("id")
        assert created_object_id is not None, "BUG: created object ID was not returned: {}".format(
            created_object
        )

        MotleyKuzuGraphStore._set_node_id(node=node, node_id=created_object_id)
        node.__graph_store__ = self
        return node

    def create_relation(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str
    ) -> None:
        """Create a relation between existing nodes.

        If the relation table does not exist, this method also creates it.

        Args:
            from_node: Source node.
            to_node: Destination node.
            label: Relation label.
        """
        assert self.check_node_exists(from_node), (
            "From-node is not present in the database, "
            "consider using upsert_triplet() for such cases"
        )
        assert self.check_node_exists(to_node), (
            "To-node is not present in the database, "
            "consider using upsert_triplet() for such cases"
        )

        self.ensure_relation_table(from_class=type(from_node), to_class=type(to_node), label=label)

        logger.info(
            "Creating relation %s from %s:%s to %s:%s",
            label,
            from_node.get_label(),
            from_node.id,
            to_node.get_label(),
            to_node.id,
        )

        create_result = self._execute_query(
            (
                "MATCH (n1:{}), (n2:{}) WHERE n1.id = $from_id AND n2.id = $to_id "
                "CREATE (n1)-[r:{}]->(n2) "
                "RETURN r"
            ).format(from_node.get_label(), to_node.get_label(), label),
            {
                "from_id": from_node.id,
                "to_id": to_node.id,
            },
        )
        assert create_result.has_next()
        logger.info("Relation created OK")

    def upsert_triplet(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str
    ) -> None:
        """Create a relation with a given label between nodes, if such does not already exist.

        If the nodes do not already exist, create them too.
        This method also creates and/or updates all necessary tables.

        Args:
            from_node: Source node.
            to_node: Destination node.
            label: Relation label.
        """
        if not self.check_node_exists(from_node):
            logger.info("Node %s does not exist, creating", from_node)
            self.insert_node(from_node)

        if not self.check_node_exists(to_node):
            logger.info("Node %s does not exist, creating", to_node)
            self.insert_node(to_node)

        if not self.check_relation_exists(from_node=from_node, to_node=to_node, label=label):
            logger.info("Relation from %s to %s does not exist, creating", from_node, to_node)
            self.create_relation(from_node=from_node, to_node=to_node, label=label)

    def delete_node(self, node: MotleyGraphNode) -> None:
        """Delete a given node and its relations.

        Args:
            node: Node to delete.
        """

        def inner_delete_relations(node_label: str, node_id: int) -> None:
            if not self.connection._get_rel_table_names():
                # Avoid Kuzu error when no relation tables exist in the database
                return

            # Undirected relation removal is not supported for some reason
            if self._check_rel_table_exists(from_label=node_label):
                self._execute_query(
                    "MATCH (n:{})-[r]->() WHERE n.id = $node_id DELETE r".format(node_label),
                    {"node_id": node_id},
                )
            if self._check_rel_table_exists(to_label=node_label):
                self._execute_query(
                    "MATCH (n:{})<-[r]-() WHERE n.id = $node_id DELETE r".format(node_label),
                    {"node_id": node_id},
                )

        def inner_delete_node(node_label: str, node_id: int) -> None:
            self._execute_query(
                "MATCH (n:{}) WHERE n.id = $node_id DELETE n".format(node_label),
                {"node_id": node_id},
            )

        assert self.check_node_exists(node), "Cannot delete nonexistent node: {}".format(node)

        inner_delete_relations(node_label=node.get_label(), node_id=node.id)
        inner_delete_node(node_label=node.get_label(), node_id=node.id)

        MotleyKuzuGraphStore._set_node_id(node, None)

    def update_property(self, node: MotleyGraphNode, property_name: str) -> MotleyGraphNode:
        """Update a graph node's property with the corresponding value from the node object.

        Args:
            node: Node to update.
            property_name: Property name to update.

        Returns:
            Updated node.
        """
        property_value = getattr(node, property_name)

        existing_property_names = self._get_node_property_names(node.get_label())

        assert property_name in node.model_fields, "No such field in node model {}: {}".format(
            node.__class__.__name__, property_name
        )

        assert self.check_node_exists(node)
        assert property_name in existing_property_names, "No such field in DB table {}: {}".format(
            node.get_label(), property_name
        )

        _, is_json = MotleyKuzuGraphStore._get_cypher_type_and_is_json_by_python_type_annotation(
            node.model_fields[property_name].annotation
        )

        db_property_name = property_name
        if is_json:
            db_property_value = MotleyKuzuGraphStore.JSON_CONTENT_PREFIX + json.dumps(
                property_value
            )
        else:
            db_property_value = property_value

        query = """
                    MATCH (n:{})
                    WHERE n.id = $node_id
                    SET n.{} = $property_value RETURN n;
                """.format(
            node.get_label(), db_property_name
        )

        query_result = self._execute_query(
            query,
            {"node_id": node.id, "property_value": db_property_value},
        )
        assert query_result.has_next()
        row = query_result.get_next()
        node_dict = row[0]
        assert node_dict[db_property_name] == db_property_value

        return node

    def run_cypher_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
        container: Optional[Type[MotleyGraphNodeType]] = None,
    ) -> list[list | MotleyGraphNodeType]:
        """Run a Cypher query and return the results.

        If container class is provided, deserialize the results into objects of that class.

        Args:
            query: Cypher query.
            parameters: Query parameters.
            container: Node class to deserialize the results into. If None, return raw results.

        Returns:
            List of query results.
        """
        query_result = self._execute_query(query=query, parameters=parameters)
        retval = []
        while query_result.has_next():
            row = query_result.get_next()
            if container is not None:
                assert len(row) == 1, "Expected single column result for deserialization"
                retval.append(self._deserialize_node(node_dict=row[0], node_class=container))
            else:
                retval.append(row)
        return retval

    def _deserialize_node(
        self, node_dict: dict, node_class: Type[MotleyGraphNode]
    ) -> MotleyGraphNode:
        """Deserialize a node from a dictionary.

        Args:
            node_dict: Dictionary representation of the node.
            node_class: Node class.

        Returns:
            Deserialized node.
        """
        for field_name, value in node_dict.copy().items():
            if isinstance(value, str) and value.startswith(
                MotleyKuzuGraphStore.JSON_CONTENT_PREFIX
            ):
                logger.debug(
                    "Value for field %s is marked as JSON, attempting to deserialize: %s",
                    field_name,
                    value,
                )
                node_dict[field_name] = json.loads(
                    value[len(MotleyKuzuGraphStore.JSON_CONTENT_PREFIX) :]
                )

        node = node_class.model_validate(node_dict)
        node._id = node_dict["id"]
        node.__graph_store__ = self
        if "id" in node_dict:
            MotleyKuzuGraphStore._set_node_id(node, node_dict["id"])
        return node

    @staticmethod
    def _set_node_id(node: MotleyGraphNode, node_id: Optional[int]) -> None:
        """Set the id of the node.

        Args:
            node: Node.
            node_id: Node id.
        """
        setattr(node, MotleyKuzuGraphStore.ID_ATTR, node_id)

    @staticmethod
    def _node_to_cypher_mapping_with_parameters(node: MotleyGraphNode) -> tuple[str, dict]:
        """Convert a node to a Cypher mapping and parameters.

        Args:
            node: Node to convert.

        Returns:
            A tuple of Cypher mapping and parameters.
        """
        node_dict = node.model_dump()

        parameters = {}

        cypher_mapping = "{"
        for field_name, value in node_dict.items():
            assert field_name != "id", "id field is reserved for node id"

            _, is_json = (
                MotleyKuzuGraphStore._get_cypher_type_and_is_json_by_python_type_annotation(
                    node.model_fields[field_name].annotation
                )
            )
            if is_json and value is not None:
                value = json.dumps(value)
                value = MotleyKuzuGraphStore.JSON_CONTENT_PREFIX + value

            cypher_mapping += f"{field_name}: ${field_name}, "
            parameters[field_name] = value

        cypher_mapping = cypher_mapping.rstrip(", ") + "}"
        return cypher_mapping, parameters

    @staticmethod
    def _get_cypher_type_and_is_json_by_python_type_annotation(
        annotation: Type,
    ) -> tuple[str, bool]:
        """Determine suitable Cypher data type by Python/Pydantic type annotation,
        and whether the data should be stored in JSON-serialized strings.

        Args:
            annotation: Python type annotation.

        Returns:
            A tuple of Cypher type and whether the data should be stored in JSON-serialized strings.
        """
        cypher_type = MotleyKuzuGraphStore.PYTHON_TO_CYPHER_TYPES_MAPPING.get(annotation)
        if not cypher_type:
            logger.info(
                "No known Cypher type matching annotation %s, will use JSON string",
                annotation,
            )
            return MotleyKuzuGraphStore.PYTHON_TO_CYPHER_TYPES_MAPPING[str], True
        return cypher_type, False

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
    ) -> "MotleyKuzuGraphStore":
        """Load from persist dir.

        Args:
            persist_dir (str): Persist directory.

        Returns:
            Graph store.
        """

        try:
            import kuzu
        except ImportError:
            raise ImportError("Please install kuzu: pip install kuzu")
        database = kuzu.Database(persist_dir)
        return cls(database)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MotleyKuzuGraphStore":
        """Initialize graph store from configuration dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Graph store.
        """
        return cls(**config_dict)
