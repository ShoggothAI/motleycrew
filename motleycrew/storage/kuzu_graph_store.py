"""
Code derived from: https://github.com/run-llama/llama_index/blob/802064aee72b03ab38ead0cda780cfa3e37ce728/llama-index-integrations/graph_stores/llama-index-graph-stores-kuzu/llama_index/graph_stores/kuzu/base.py
Kùzu graph store index.
"""

import logging
from time import sleep

from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel
import kuzu
from kuzu import PreparedStatement, QueryResult
import json

from graph_store import MotleyGraphStore


ModelType = TypeVar("ModelType", bound=BaseModel)


class RetryConnection(kuzu.Connection):
    def execute(self, *args, **kwargs) -> QueryResult:
        sleep(0)
        return super().execute(*args, **kwargs)


class MotleyKuzuGraphStore(MotleyGraphStore):
    TABLE_NAME_ATTR = "__tablename__"
    ID_ATTR = "_id"

    RELATION_TABLE_NAME_TEMPLATE = "FROM_{src}_TO_{dst}"
    JSON_FIELD_PREFIX = "JSON__"

    PYTHON_TO_CYPHER_TYPES_MAPPING = {
        int: "INT64",  # TODO: enforce size when creating and updating entities and relations
        Optional[int]: "INT64",
        str: "STRING",
        Optional[str]: "STRING",
        float: "DOUBLE",
        Optional[float]: "DOUBLE",
        bool: "BOOLEAN",
        Optional[bool]: "BOOLEAN",
    }

    def __init__(self, database: Any) -> None:
        self.database = database
        self.connection = RetryConnection(database)

    def _execute_query(
        self, query: str | PreparedStatement, parameters: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """
        Execute a query, logging it for debugging purposes
        """
        logging.debug("Executing query: %s", query)
        if parameters:
            logging.debug("with parameters: %s", parameters)

        return self.connection.execute(query=query, parameters=parameters)

    def _check_node_table_exists(self, table_name: str):
        return table_name in self.connection._get_node_table_names()

    def _check_rel_table_exists(self, table_name: str):
        return table_name in [row["name"] for row in self.connection._get_rel_table_names()]

    def _get_node_property_names(self, table_name: str):
        return self.connection._get_node_property_names(table_name=table_name)

    def _ensure_entity_table(self, entity: ModelType) -> str:
        """
        Create a table for storing entities of that class if such does not already exist.
        If it does exist, create all missing columns.
        The table name is inferred from the __tablename__ attribute if it is set,
        otherwise from the class name.
        Return the table name.
        """
        table_name = MotleyKuzuGraphStore._get_entity_table_name(entity)
        if not self._check_node_table_exists(table_name):
            logging.info("Node table %s does not exist in the database, creating", table_name)
            self._execute_query(
                "CREATE NODE TABLE {} (id SERIAL, PRIMARY KEY(id))".format(table_name)
            )

        # Create missing property columns
        existing_property_names = self._get_node_property_names(table_name=table_name)
        for field_name, field in entity.model_fields.items():
            if (
                field_name not in existing_property_names
                and MotleyKuzuGraphStore.JSON_FIELD_PREFIX + field_name
                not in existing_property_names
            ):
                logging.info("Property %s not present in table %s, creating")
                assert not field_name.startswith(MotleyKuzuGraphStore.JSON_FIELD_PREFIX)
                cypher_type, is_json = (
                    MotleyKuzuGraphStore._get_cypher_type_and_is_json_by_python_type_annotation(
                        field.annotation
                    )
                )
                if is_json:
                    field_name = MotleyKuzuGraphStore.JSON_FIELD_PREFIX + field_name

                self._execute_query(
                    "ALTER TABLE {} ADD {} {}".format(table_name, field_name, cypher_type)
                )
        return table_name

    def _ensure_relation_table(self, from_entity: ModelType, to_entity: ModelType) -> str:
        """
        Create a table for storing relations from from_entity-like nodes to to_entity-like nodes,
        if such does not already exist.
        Return the table name.
        """
        table_name = MotleyKuzuGraphStore._get_relation_table_name(
            from_entity=from_entity, to_entity=to_entity
        )
        if not self._check_rel_table_exists(table_name):
            logging.info("Relation table %s does not exist in the database, creating", table_name)

            from_table_name = MotleyKuzuGraphStore._get_entity_table_name(from_entity)
            to_table_name = MotleyKuzuGraphStore._get_entity_table_name(to_entity)
            self._execute_query(
                "CREATE REL TABLE {} (FROM {} TO {}, predicate STRING)".format(
                    table_name, from_table_name, to_table_name
                )
            )
        return table_name

    def check_entity_exists_by_class_and_id(
        self, entity_class: Type[ModelType], entity_id: int
    ) -> bool:
        """
        Check if an entity of given class with given id is present in the database.
        """
        table_name = MotleyKuzuGraphStore._get_entity_table_name_by_entity_class(entity_class)
        if not self._check_node_table_exists(table_name):
            return False

        is_exists_result = self._execute_query(
            "MATCH (n:{}) WHERE n.id = $entity_id RETURN n.id".format(table_name),
            {"entity_id": entity_id},
        )
        return is_exists_result.has_next()

    def check_entity_exists(self, entity: ModelType) -> bool:
        """
        Check if the given entity is present in the database.
        """
        entity_id = MotleyKuzuGraphStore._get_entity_id(entity)
        if entity_id is None:
            return False  # for cases when id attribute is not set => entity does not exist

        return self.check_entity_exists_by_class_and_id(
            entity_class=entity.__class__, entity_id=entity_id
        )

    def check_relation_exists(
        self, from_entity: ModelType, to_entity: ModelType, predicate: Optional[str] = None
    ) -> bool:
        """
        Check if a relation exists between two entities with given predicate.
        """
        from_entity_id = MotleyKuzuGraphStore._get_entity_id(from_entity)
        to_entity_id = MotleyKuzuGraphStore._get_entity_id(to_entity)
        assert (
            from_entity_id is not None and to_entity_id is not None
        ), "Both entities ids must be set"

        from_table_name = MotleyKuzuGraphStore._get_entity_table_name(from_entity)
        to_table_name = MotleyKuzuGraphStore._get_entity_table_name(to_entity)
        relation_table_name = MotleyKuzuGraphStore._get_relation_table_name(
            from_entity=from_entity, to_entity=to_entity
        )
        if (
            not self._check_node_table_exists(from_table_name)
            or not self._check_node_table_exists(to_table_name)
            or not self._check_rel_table_exists(relation_table_name)
        ):
            return False

        query = (
            "MATCH (n1:{})-[r:{}]->(n2:{}) "
            "WHERE n1.id = $from_entity_id AND n2.id = $to_entity_id {}"
            "RETURN r".format(
                from_table_name,
                relation_table_name,
                to_table_name,
                "AND r.predicate = $predicate " if predicate is not None else "",
            )
        )
        parameters = {
            "from_entity_id": from_entity_id,
            "to_entity_id": to_entity_id,
        }
        if predicate is not None:
            parameters["predicate"] = predicate

        is_exists_result = self._execute_query(query=query, parameters=parameters)
        return is_exists_result.has_next()

    def get_entity_by_class_and_id(
        self, entity_class: Type[ModelType], entity_id: int
    ) -> Optional[ModelType]:
        """
        Retrieve the entity of given class with given id if it is present in the database.
        Otherwise, return None.
        """
        table_name = MotleyKuzuGraphStore._get_entity_table_name_by_entity_class(entity_class)
        if not self._check_node_table_exists(table_name):
            return None

        query = """
                    MATCH (n:{})
                    WHERE n.id = $entity_id
                    RETURN n;
                """.format(
            table_name
        )
        query_result = self._execute_query(query, {"entity_id": entity_id})

        if query_result.has_next():
            row = query_result.get_next()
            entity_dict = row[0]
            for field_name, value in entity_dict.copy().items():
                if (
                    field_name.startswith(MotleyKuzuGraphStore.JSON_FIELD_PREFIX)
                    and value is not None
                ):
                    logging.debug(
                        "Field %s is marked as JSON, attempting to deserialize value: %s",
                        field_name,
                        value,
                    )
                    new_field_name = field_name[len(MotleyKuzuGraphStore.JSON_FIELD_PREFIX) :]
                    entity_dict[new_field_name] = json.loads(value)
                    entity_dict.pop(field_name)

            return entity_class.parse_obj(entity_dict)

    def create_entity(self, entity: ModelType) -> ModelType:
        """
        Create a new entity, populate its id and freeze it.
        If entity table or some columns do not exist, this method also creates them.
        """
        assert not MotleyKuzuGraphStore._get_entity_id(
            entity
        ), "Entity has its {} attribute set, looks like it is already in the DB".format(
            MotleyKuzuGraphStore.ID_ATTR
        )

        table_name = self._ensure_entity_table(entity)
        logging.info("Creating entity in table %s: %s", table_name, entity)

        cypher_mapping, parameters = MotleyKuzuGraphStore._entity_to_cypher_mapping_with_parameters(
            entity
        )
        create_result = self._execute_query(
            "CREATE (n:{} {}) RETURN n".format(table_name, cypher_mapping),
            parameters=parameters,
        )
        assert create_result.has_next()
        logging.info("Entity created OK")

        created_object = create_result.get_next()[0]
        created_object_id = created_object.get("id")
        assert created_object_id is not None, "BUG: created object ID was not returned: {}".format(
            created_object
        )

        MotleyKuzuGraphStore._set_entity_id(entity=entity, entity_id=created_object_id)
        MotleyKuzuGraphStore._freeze_entity(entity)
        return entity

    def create_relation(self, from_entity: ModelType, to_entity: ModelType, predicate: str) -> None:
        """
        Create a relation between existing entities.
        If relation table does not exist, this method also creates them.
        """
        assert self.check_entity_exists(from_entity), (
            "From-entity is not present in the database, "
            "consider using upsert_triplet() for such cases"
        )
        assert self.check_entity_exists(to_entity), (
            "To-entity is not present in the database, "
            "consider using upsert_triplet() for such cases"
        )

        table_name = self._ensure_relation_table(from_entity=from_entity, to_entity=to_entity)
        from_table_name = self._get_entity_table_name(from_entity)
        to_table_name = self._get_entity_table_name(to_entity)
        from_entity_id = MotleyKuzuGraphStore._get_entity_id(from_entity)
        to_entity_id = MotleyKuzuGraphStore._get_entity_id(to_entity)

        logging.info(
            "Creating relation %s from %s:%s to %s:%s",
            predicate,
            from_table_name,
            from_entity_id,
            to_table_name,
            to_entity_id,
        )

        create_result = self._execute_query(
            (
                "MATCH (n1:{}), (n2:{}) WHERE n1.id = $from_id AND n2.id = $to_id "
                "CREATE (n1)-[r:{} {{predicate: $predicate}}]->(n2) "
                "RETURN r"
            ).format(from_table_name, to_table_name, table_name),
            {
                "from_id": from_entity_id,
                "to_id": to_entity_id,
                "predicate": predicate,
            },
        )
        assert create_result.has_next()
        logging.info("Relation created OK")

    def upsert_triplet(self, from_entity: ModelType, to_entity: ModelType, predicate: str):
        """
        Create a relation with a given predicate between entities, if such does not already exist.
        If the entities do not already exist, create them too.
        This method also creates and/or updates all necessary tables.
        """
        if not self.check_entity_exists(from_entity):
            logging.info("Entity %s does not exist, creating", from_entity)
            self.create_entity(from_entity)

        if not self.check_entity_exists(to_entity):
            logging.info("Entity %s does not exist, creating", to_entity)
            self.create_entity(to_entity)

        if not self.check_relation_exists(
            from_entity=from_entity, to_entity=to_entity, predicate=predicate
        ):
            logging.info("Relation from %s to %s does not exist, creating", from_entity, to_entity)
            self.create_relation(from_entity=from_entity, to_entity=to_entity, predicate=predicate)

    def delete_entity(self, entity: ModelType) -> None:
        """
        Delete a given entity and its relations.
        """

        def inner_delete_relations(table_name: str, entity_id: int) -> None:
            # Undirected relation removal is not supported for some reason
            self._execute_query(
                "MATCH (n:{})-[r]->() WHERE n.id = $entity_id DELETE r".format(table_name),
                {"entity_id": entity_id},
            )
            self._execute_query(
                "MATCH (n:{})<-[r]-() WHERE n.id = $entity_id DELETE r".format(table_name),
                {"entity_id": entity_id},
            )

        def inner_delete_entity(table_name: str, entity_id: int) -> None:
            self._execute_query(
                "MATCH (n:{}) WHERE n.id = $entity_id DELETE n".format(table_name),
                {"entity_id": entity_id},
            )

        assert self.check_entity_exists(entity), "Cannot delete nonexistent entity: {}".format(
            entity
        )

        table_name = MotleyKuzuGraphStore._get_entity_table_name(entity)
        entity_id = MotleyKuzuGraphStore._get_entity_id(entity)
        inner_delete_relations(table_name=table_name, entity_id=entity_id)
        inner_delete_entity(table_name=table_name, entity_id=entity_id)

        MotleyKuzuGraphStore._unfreeze_entity(entity)
        setattr(entity, MotleyKuzuGraphStore.ID_ATTR, None)

    def set_property(self, entity: ModelType, property_name: str, property_value: Any) -> None:
        """
        Set a property to an entity. Also sets the property in the Python object.
        """
        entity_id = MotleyKuzuGraphStore._get_entity_id(entity)
        table_name = MotleyKuzuGraphStore._get_entity_table_name(entity)
        existing_property_names = self._get_node_property_names(table_name=table_name)

        assert property_name in entity.model_fields, "No such field in Pydantic model: {}".format(
            property_name
        )

        assert self.check_entity_exists(entity)
        assert (
            property_name in existing_property_names
            or MotleyKuzuGraphStore.JSON_FIELD_PREFIX + property_name in existing_property_names
        ), "No such field in DB table {}: {}".format(table_name, property_name)

        # Running Pydantic validation beforehand to avoid writing invalid values to the DB
        entity.__pydantic_validator__.validate_assignment(
            entity.model_construct(), property_name, property_value
        )

        if MotleyKuzuGraphStore.JSON_FIELD_PREFIX + property_name in existing_property_names:
            db_property_name = MotleyKuzuGraphStore.JSON_FIELD_PREFIX + property_name
            db_property_value = json.dumps(property_value) if property_value is not None else None
        else:
            db_property_name = property_name
            db_property_value = property_value

        query = """
                    MATCH (n:{})
                    WHERE n.id = $entity_id
                    SET n.{} = $property_value RETURN n;
                """.format(
            table_name, db_property_name
        )

        query_result = self._execute_query(
            query,
            {"entity_id": entity_id, "property_value": db_property_value},
        )
        assert query_result.has_next()
        row = query_result.get_next()
        entity_dict = row[0]
        assert entity_dict[db_property_name] == db_property_value

        # Now set the property value in the Python object
        MotleyKuzuGraphStore._unfreeze_entity(entity)
        setattr(entity, property_name, property_value)
        MotleyKuzuGraphStore._freeze_entity(entity)
        return entity

    def run_cypher_query(self, query: str, parameters: Optional[dict] = None) -> list[list]:
        """
        Run a Cypher query and return the results.
        """
        query_result = self._execute_query(query=query, parameters=parameters)
        retval = []
        while query_result.has_next():
            retval.append(query_result.get_next())
        return retval

    @staticmethod
    def _get_entity_table_name(entity: ModelType) -> str:
        table_name = getattr(entity, MotleyKuzuGraphStore.TABLE_NAME_ATTR, None)
        if not table_name:
            table_name = entity.__class__.__name__.lower()

        return table_name

    @staticmethod
    def _get_entity_table_name_by_entity_class(entity_class: Type[ModelType]) -> str:
        table_name = getattr(entity_class, MotleyKuzuGraphStore.TABLE_NAME_ATTR, None)
        if not table_name:
            table_name = entity_class.__name__.lower()

        return table_name

    @staticmethod
    def _get_relation_table_name(from_entity: ModelType, to_entity: ModelType) -> str:
        from_entity_table_name = MotleyKuzuGraphStore._get_entity_table_name(from_entity)
        to_entity_table_name = MotleyKuzuGraphStore._get_entity_table_name(to_entity)

        return MotleyKuzuGraphStore.RELATION_TABLE_NAME_TEMPLATE.format(
            src=from_entity_table_name, dst=to_entity_table_name
        )

    @staticmethod
    def _get_entity_id(entity: ModelType) -> Optional[int]:
        return getattr(entity, MotleyKuzuGraphStore.ID_ATTR, None)

    @staticmethod
    def _set_entity_id(entity: ModelType, entity_id: int) -> None:
        setattr(entity, MotleyKuzuGraphStore.ID_ATTR, entity_id)

    @staticmethod
    def _freeze_entity(entity: ModelType) -> None:
        """
        Make the entity immutable by enabling its model_config["frozen"].
        Depends on the corresponding Pydantic feature.
        See https://docs.pydantic.dev/latest/concepts/models/#faux-immutability
        """
        assert (
            MotleyKuzuGraphStore._get_entity_id(entity) is not None
        ), "Cannot freeze entity because its id is not set, it may not be in the database yet"

        entity.model_config["frozen"] = True

    @staticmethod
    def _unfreeze_entity(entity: ModelType) -> None:
        """
        Reverse operation to _freeze_entity().
        """
        entity.model_config["frozen"] = False

    @staticmethod
    def _entity_to_cypher_mapping_with_parameters(entity: ModelType) -> tuple[str, dict]:
        entity_dict = entity.model_dump()

        parameters = {}

        cypher_mapping = "{"
        for field_name, value in entity_dict.items():
            if value is None:
                # TODO: remove (as of Kuzu v0.3.2, parameters of type NoneType are not supported)
                continue

            _, is_json = (
                MotleyKuzuGraphStore._get_cypher_type_and_is_json_by_python_type_annotation(
                    entity.model_fields[field_name].annotation
                )
            )
            if is_json:
                field_name = MotleyKuzuGraphStore.JSON_FIELD_PREFIX + field_name
                if value is not None:
                    value = json.dumps(value)

            cypher_mapping += f"{field_name}: ${field_name}, "
            parameters[field_name] = value

        cypher_mapping = cypher_mapping.rstrip(", ") + "}"
        return cypher_mapping, parameters

    @staticmethod
    def _get_cypher_type_and_is_json_by_python_type_annotation(
        annotation: Type,
    ) -> tuple[str, bool]:
        """
        Determine suitable Cypher data type by Python/Pydantic type annotation,
        and whether the data should be stored in JSON-serialized strings.
        """
        cypher_type = MotleyKuzuGraphStore.PYTHON_TO_CYPHER_TYPES_MAPPING.get(annotation)
        if not cypher_type:
            logging.warning(
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
        """Load from persist dir."""
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


if __name__ == "__main__":
    from pathlib import Path
    import shutil

    here = Path(__file__).parent
    db_path = here / "test1"
    shutil.rmtree(db_path, ignore_errors=True)
    db = kuzu.Database(str(db_path))
    graph_store = MotleyKuzuGraphStore(db)

    class Question(BaseModel):
        question: str
        answer: Optional[str] = None
        context: Optional[List[str]] = None

    q1 = Question(question="q1")
    graph_store.create_entity(q1)
    assert getattr(q1, "_id", None) is not None
    q1_id = q1._id

    assert graph_store.check_entity_exists(q1)
    assert graph_store.check_entity_exists_by_class_and_id(entity_class=Question, entity_id=q1_id)

    q2 = Question(question="q2", answer="a2")
    graph_store.upsert_triplet(from_entity=q1, to_entity=q2, predicate="p")
    assert getattr(q2, "_id", None) is not None
    q2_id = q2._id

    assert graph_store.check_relation_exists(from_entity=q1, to_entity=q2, predicate="p")
    assert not graph_store.check_relation_exists(from_entity=q2, to_entity=q1)

    graph_store.delete_entity(q1)
    assert not graph_store.check_entity_exists(q1)
    assert graph_store.get_entity_by_class_and_id(entity_class=Question, entity_id=q1_id) is None

    graph_store.set_property(q2, property_name="context", property_value=["abc", "def"])
    assert q2.context == ["abc", "def"]

    assert graph_store.get_entity_by_class_and_id(
        entity_class=Question, entity_id=q2_id
    ).context == ["abc", "def"]

    print(f"docker run -p 8000:8000  -v {db_path}:/database --rm kuzudb/explorer: latest")
    print("MATCH (A)-[r]->(B) RETURN *;")
