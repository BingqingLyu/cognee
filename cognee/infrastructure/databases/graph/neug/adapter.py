"""NeuG graph adapter.

Ports the Ladybug (Kuzu-family) single-table schema onto NeuG, an embedded
Kuzu-dialect graph database: one ``Node`` table (``id`` STRING primary key +
``type`` column + JSON ``properties`` blob) and one ``EDGE`` rel table
(``relationship_name`` column). This shape avoids dynamic labels, dynamic
relationship types and APOC entirely.

Dialect decisions come from the cognee dialect probe
(``neug-memory-benchmark/probes/probe_neug_cognee.py``):

- MERGE keyed on the primary key with ON CREATE/ON MATCH SET is the upsert
  path; MERGE map literals must only contain the PK (functions inside the
  map literal trigger an internal error).
- ``IN $list`` parameters segfault NeuG 0.2.0 — every multi-id filter is
  built as a parameterized OR-chain instead.
- TIMESTAMP columns accept ISO strings as parameters directly (the
  ``TIMESTAMP($param)`` function form is rejected).
- RETURN map literals are unsupported — all queries return flat columns and
  rows are assembled into dicts on the Python side.
- ``type(r)``/``keys(n)``/``properties(n)`` do not exist; ``query()`` shims
  them for Cypher pass-through (see ``_adapt_cypher``).

The adapter does not own the database connection: graph and vector adapters
share one read-write NeuG database through the process-level
``NeuGConnectionManager`` (single rw connection, statement-level asyncio
serialization, refcounted close).
"""

import json
import re
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.neug import get_neug_connection_manager
from cognee.infrastructure.engine import DataPoint
from cognee.modules.retrieval.natural_language_retriever import GuidanceSchemaRow
from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_logger

logger = get_logger("NeuGGraphAdapter")

# Progress logging granularity for per-row write loops (NeuG has no UNWIND;
# writes are executed as individual parameterized MERGE statements).
_WRITE_CHUNK_SIZE = 256

# ``type(x)`` does not exist in NeuG; ``label(x)`` returns the table name and
# is the closest equivalent for LLM-generated Cypher.
_TYPE_CALL_RE = re.compile(r"\btype\s*\(")


def _adapt_cypher_query(query: str) -> str:
    """Rewrite ``type(x)`` calls to ``label(x)`` outside string literals.

    ``type()`` does not exist in NeuG; ``label()`` returns the table name
    and is the closest equivalent for LLM-generated Cypher. Single-quoted
    segments are skipped so literals like ``'my type (test)'`` keep their
    text (Cypher escapes quotes by doubling, which stays inside one
    segment pair).
    """
    parts = query.split("'")
    for index in range(0, len(parts), 2):
        parts[index] = _TYPE_CALL_RE.sub("label(", parts[index])
    return "'".join(parts)


# NeuG reports unknown Cypher labels as ``Schema mismatch: Table X does not
# exist``. LLM-generated Cypher (NATURAL_LANGUAGE search) assumes cognee's
# per-type labels (``:Entity``, ``:EntityType``, ...), which do not exist in
# this adapter's single-table schema; the fallback below rewrites the missing
# label onto the Node/EDGE tables instead.
_MISSING_TABLE_RE = re.compile(r"Table (\w+) does not exist")
# NeuG names one missing table per error; a query can reference several
# unknown labels, so allow one rewrite round per distinct label, capped.
_MAX_LABEL_FALLBACK_ROUNDS = 8

# NeuG prefixes result column names per binding block (``_0_n.name``); the
# pass-through row shaping strips the prefix so dict rows carry readable keys.
_COLUMN_PREFIX_RE = re.compile(r"^_\d+_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def _parse_properties_blob(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class NeuGGraphAdapter(GraphDBInterface):
    """GraphDBInterface implementation backed by the embedded NeuG database."""

    supports_cypher_queries = True

    def __init__(
        self,
        graph_database_url: str = "",
        graph_database_username: str = "",
        graph_database_password: str = "",
        graph_database_port: int = 0,
        graph_database_key: str = "",
        database_name: str = "",
    ):
        """The connection parameters are accepted for factory compatibility but
        unused: the NeuG database path is resolved centrally by the shared
        connection manager (``NEUG_DB_PATH`` env var, or the cognee data root)."""
        self.connection_manager = get_neug_connection_manager()
        self.connection_manager.acquire()
        self._closed = False
        self._schema_ensured = False

    # ------------------------------------------------------------------
    # Schema / execution plumbing
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        if self._schema_ensured:
            return
        await self.connection_manager.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Node(
                id STRING PRIMARY KEY,
                name STRING,
                type STRING,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                properties VARCHAR(65535)
            )
            """
        )
        await self.connection_manager.execute(
            """
            CREATE REL TABLE IF NOT EXISTS EDGE(
                FROM Node TO Node,
                relationship_name STRING,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                properties VARCHAR(65535)
            )
            """
        )
        self._schema_ensured = True

    async def _execute(self, query: str, params: Optional[dict] = None) -> List[List[Any]]:
        await self.ensure_schema()
        return await self.connection_manager.execute(query, params)

    async def _execute_with_columns(
        self, query: str, params: Optional[dict] = None
    ) -> Tuple[List[List[Any]], List[str]]:
        await self.ensure_schema()
        return await self.connection_manager.execute_with_columns(query, params)

    @staticmethod
    def _ids_where_clause(alias: str, ids: List[str], params: dict, prefix: str = "nid") -> str:
        """Build a parameterized OR-chain for ``alias.id IN ids``.

        NeuG 0.2.0 crashes on list parameters (``IN $ids``), so multi-id
        filters are always expanded into equality comparisons.
        """
        clauses = []
        for i, node_id in enumerate(ids):
            param = f"{prefix}_{i}"
            params[param] = str(node_id)
            clauses.append(f"{alias}.id = ${param}")
        return "(" + " OR ".join(clauses) + ")"

    @staticmethod
    def _node_dict(node_id, name, node_type, properties_blob) -> Dict[str, Any]:
        data = {"id": node_id, "name": name, "type": node_type}
        data.update(_parse_properties_blob(properties_blob))
        return data

    _NODE_COLUMNS = "n.id, n.name, n.type, n.properties"

    # ------------------------------------------------------------------
    # Cypher pass-through (CYPHER / NATURAL_LANGUAGE search types)
    # ------------------------------------------------------------------

    def _adapt_cypher(self, query: str) -> str:
        return _adapt_cypher_query(query)

    @staticmethod
    def _rewrite_missing_label(query: str, error: str) -> Optional[str]:
        """Rewrite a missing Cypher label onto the single-table schema.

        Returns the rewritten query when the error names a table that is
        neither of this adapter's own tables, else ``None`` (no fallback).
        The error's binder site tells which position failed: rel-table
        errors rewrite every relationship-pattern label (with or without a
        bound variable) onto ``EDGE``; node-table errors rewrite node
        labels onto ``Node``.
        """
        match = _MISSING_TABLE_RE.search(error)
        if not match:
            return None
        missing = match.group(1)
        if missing in ("Node", "EDGE"):
            return None
        if "bindRelTableEntries" in error or "relationship pattern label" in error:
            # Relationship-pattern labels: with a bound variable ([r:Label])
            # or bare ([:Label]); the optional group keeps either prefix.
            rewritten = re.sub(
                rf"\[(\w+:|:)?{re.escape(missing)}\b",
                lambda m: "[" + (m.group(1) or "") + "EDGE",
                query,
            )
        else:
            rewritten = re.sub(rf":{re.escape(missing)}\b", ":Node", query)
        return rewritten if rewritten != query else None

    @staticmethod
    def _lowercase_string_literals(query: str) -> str:
        """Lowercase single-quoted string literals in generated Cypher.

        cognee stores entity/type names lowercase; LLMs habitually emit
        capitalized literals ('Person'), which then match nothing. Applied
        only as a last-chance retry after a fallback-rewritten query ran
        clean but returned no rows.
        """
        return re.sub(r"'([^']*)'", lambda m: "'" + m.group(1).lower() + "'", query)

    @staticmethod
    def _normalize_cell(value: Any) -> Any:
        """Convert a raw NeuG result cell into a cognee-friendly value.

        Node cells from ``RETURN n`` carry NeuG's internal ``_ID``/``_LABEL``
        keys and a serialized ``properties`` blob; reshape them like
        ``_node_dict`` so consumers see the same flattened node dicts as on
        the Neo4j/Ladybug Cypher paths.
        """
        if isinstance(value, dict) and "_LABEL" in value:
            node = {key: val for key, val in value.items() if key not in ("_ID", "_LABEL")}
            node.update(_parse_properties_blob(node.pop("properties", None)))
            return node
        return value

    @classmethod
    def _shape_rows(cls, rows: List[List[Any]], columns: List[str]) -> List[Any]:
        """Shape raw NeuG rows into the dict-row contract of other adapters.

        ``SearchResultPayload`` validates CYPHER / NATURAL_LANGUAGE results
        as str / list[str] / list[dict] / dict / BaseModel, so flat
        multi-column rows must become dicts; single-column rows are unwrapped
        to scalars so they validate as ``list[str]``-like results.
        """
        if not rows:
            return rows
        names = [_COLUMN_PREFIX_RE.sub("", col) for col in columns or []]
        if len(names) == 1:
            return [cls._normalize_cell(row[0]) for row in rows]
        if names and all(len(row) == len(names) for row in rows):
            return [
                {name: cls._normalize_cell(value) for name, value in zip(names, row)}
                for row in rows
            ]
        return [[cls._normalize_cell(value) for value in row] for row in rows]

    def _static_schema_introspection(self, query: str) -> List[List[Any]]:
        """Answer cognee's ``UNWIND keys(...)`` schema-introspection queries.

        NeuG has no ``keys()`` function; with this adapter's fixed two-table
        schema the answer is known statically, so return it directly instead
        of executing the query. An extra guidance row is appended because the
        natural-language prompt hardcodes cognee semantic types (Entity,
        EntityType, ...) as *labels* while on this single-table schema they
        are values of the ``type`` column — without the hint the LLM keeps
        generating ``WHERE n.type = 'person'`` style mismatches.
        """
        if "labels(" in query.lower():
            # Node schema row: (labels, property keys) as the natural-language
            # retriever's ``RETURN DISTINCT labels(n) AS NodeLabels,
            # collect(DISTINCT prop) AS Properties`` would produce it.
            return [
                [["Node"], ["id", "name", "type", "created_at", "updated_at", "properties"]],
                # Sentinel row: ``_format_node_schemas`` renders it as
                # free-form guidance (a length-based heuristic cannot tell
                # guidance apart from a real single-property label).
                GuidanceSchemaRow(
                    "single-table schema: the only node label is Node; cognee "
                    "semantic types (Entity, EntityType, DocumentChunk, "
                    "TextSummary, ...) are VALUES of the n.type column, never "
                    "node names; node names are ALL LOWERCASE; kind nodes "
                    "like 'person'/'organization' are EDGE TARGETS: instances "
                    "point TO them, so filter the target, e.g. "
                    "MATCH (e:Node)-[:EDGE]->(t:Node) WHERE t.name = 'person' "
                    "RETURN e.name (WRONG, returns nothing: "
                    "WHERE p.name = 'person' ... RETURN t.name)"
                ),
            ]
        # Edge schema: rendered verbatim into the natural-language prompt's
        # ``Edge schema`` section, so return readable single-table guidance
        # (every relationship lives in one EDGE table whose
        # ``relationship_name`` column carries the semantic type).
        return [
            GuidanceSchemaRow(
                "single EDGE relationship table: MATCH (a:Node)-[r:EDGE]->(b:Node); "
                "the semantic relationship type is the r.relationship_name column "
                "(e.g. 'mentioned_in', 'is_type_of'), not the edge label"
            )
        ]

    async def query(self, query: str, params: Optional[dict] = None) -> List[Any]:
        """Execute raw Cypher against NeuG.

        Three compatibility shims are applied: ``keys(...)`` introspection
        queries are answered statically (the function does not exist in NeuG),
        ``type(x)`` calls are rewritten to ``label(x)``, and labels that do
        not exist as tables are rewritten onto the single Node/EDGE schema
        (single-table fallback for LLM-generated Cypher). NeuG reports one
        missing table per error, so the fallback loops until the query binds
        or no further rewrite applies. Rows are shaped to the dict-row
        contract of the other Cypher adapters (see ``_shape_rows``).
        """
        if "unwind keys(" in query.lower():
            return self._static_schema_introspection(query)
        current = query
        columns: List[str] = []
        for _ in range(_MAX_LABEL_FALLBACK_ROUNDS):
            try:
                results, columns = await self._execute_with_columns(
                    self._adapt_cypher(current), params
                )
                break
            except Exception as e:
                rewritten = self._rewrite_missing_label(current, str(e))
                if rewritten is None:
                    raise
                logger.warning(
                    "Cypher label fallback: rewriting unknown label onto Node/EDGE "
                    "single-table schema (%s); rewritten query: %.300s",
                    str(e)[:200],
                    rewritten,
                )
                current = rewritten
        else:
            results, columns = await self._execute_with_columns(self._adapt_cypher(current), params)
        if not results and current != query:
            # The label-fallback rewrite bound cleanly but matched nothing.
            # Typical cause: the LLM capitalized name literals ('Person')
            # while cognee stores them lowercase. One last retry with
            # lowercased string literals before giving up.
            lowered = self._lowercase_string_literals(current)
            if lowered != current:
                logger.warning(
                    "Cypher literal-case fallback: retrying rewritten query with "
                    "lowercased string literals: %.300s",
                    lowered,
                )
                results, columns = await self._execute_with_columns(
                    self._adapt_cypher(lowered), params
                )
                current = lowered
        if not results and current != query:
            logger.warning("Cypher fallback chain returned no rows; final query: %.300s", current)
        return self._shape_rows(results, columns)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def is_empty(self) -> bool:
        rows = await self._execute("MATCH (n:Node) RETURN COUNT(n) > 0")
        return not (rows and rows[0][0])

    async def has_node(self, node_id: str) -> bool:
        rows = await self._execute(
            "MATCH (n:Node) WHERE n.id = $id RETURN COUNT(n) > 0", {"id": str(node_id)}
        )
        return bool(rows and rows[0][0])

    def _node_merge_params(
        self, node: Union[DataPoint, str], properties: Optional[dict] = None
    ) -> dict:
        if isinstance(node, str):
            # Interface contract: a string node id with an explicit property
            # dict (``add_node("id", {...})``), used e.g. by
            # ``add_model_class_to_graph``.
            properties = {"id": node, **(properties or {})}
        else:
            properties = node.model_dump() if hasattr(node, "model_dump") else vars(node)
        core = {
            "id": str(properties.get("id", "")),
            "name": str(properties.get("name", "")),
            "type": str(properties.get("type", "")),
        }
        for key in core:
            properties.pop(key, None)
        now = _now_iso()
        return {
            "param_id": core["id"],
            "name": core["name"],
            "type": core["type"],
            "properties": json.dumps(properties, cls=JSONEncoder),
            "created_at": now,
            "updated_at": now,
        }

    _NODE_MERGE_QUERY = """
        MERGE (n:Node {id: $param_id})
        ON CREATE SET
            n.name = $name,
            n.type = $type,
            n.properties = $properties,
            n.created_at = $created_at,
            n.updated_at = $updated_at
        ON MATCH SET
            n.name = $name,
            n.type = $type,
            n.properties = $properties,
            n.updated_at = $updated_at
    """

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[dict] = None
    ) -> None:
        try:
            await self._execute(self._NODE_MERGE_QUERY, self._node_merge_params(node, properties))
        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            raise

    async def add_nodes(
        self,
        nodes: List[DataPoint],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        # Graph provenance is intentionally not implemented; the write path
        # falls back to cognee's relational rollback ledger.
        if not nodes:
            return
        try:
            total = len(nodes)
            for index, node in enumerate(nodes):
                await self._execute(self._NODE_MERGE_QUERY, self._node_merge_params(node))
                if total > _WRITE_CHUNK_SIZE and (index + 1) % _WRITE_CHUNK_SIZE == 0:
                    logger.info("Merged nodes %d/%d", index + 1, total)
        except Exception as e:
            logger.error(f"Failed to add nodes in batch: {e}")
            raise

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        rows = await self._execute(
            f"MATCH (n:Node) WHERE n.id = $id RETURN {self._NODE_COLUMNS}",
            {"id": str(node_id)},
        )
        if not rows:
            return None
        return self._node_dict(*rows[0])

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        if not node_ids:
            return []
        params: dict = {}
        where = self._ids_where_clause("n", node_ids, params)
        rows = await self._execute(
            f"MATCH (n:Node) WHERE {where} RETURN {self._NODE_COLUMNS}", params
        )
        return [self._node_dict(*row) for row in rows]

    async def extract_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return await self.get_node(node_id)

    async def extract_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        return await self.get_nodes(node_ids)

    async def delete_node(self, node_id: str) -> None:
        await self._execute("MATCH (n:Node) WHERE n.id = $id DETACH DELETE n", {"id": str(node_id)})

    async def delete_nodes(self, node_ids: List[str]) -> None:
        if not node_ids:
            return
        params: dict = {}
        where = self._ids_where_clause("n", node_ids, params)
        await self._execute(f"MATCH (n:Node) WHERE {where} DETACH DELETE n", params)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    _EDGE_MERGE_QUERY = """
        MATCH (f:Node {id: $from_id}), (t:Node {id: $to_id})
        MERGE (f)-[r:EDGE {relationship_name: $relationship_name}]->(t)
        ON CREATE SET
            r.created_at = $created_at,
            r.updated_at = $updated_at,
            r.properties = $properties
        ON MATCH SET
            r.updated_at = $updated_at,
            r.properties = $properties
    """

    def _edge_merge_params(
        self, from_node: str, to_node: str, relationship_name: str, properties: Dict[str, Any]
    ) -> dict:
        now = _now_iso()
        return {
            "from_id": str(from_node),
            "to_id": str(to_node),
            "relationship_name": relationship_name,
            "created_at": now,
            "updated_at": now,
            "properties": json.dumps(properties or {}, cls=JSONEncoder),
        }

    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Dict[str, Any] = {},
    ) -> None:
        try:
            await self._execute(
                self._EDGE_MERGE_QUERY,
                self._edge_merge_params(from_node, to_node, relationship_name, edge_properties),
            )
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            raise

    async def add_edges(
        self,
        edges: List[Tuple[str, str, str, Dict[str, Any]]],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        if not edges:
            return
        try:
            total = len(edges)
            for index, (from_node, to_node, relationship_name, properties) in enumerate(edges):
                await self._execute(
                    self._EDGE_MERGE_QUERY,
                    self._edge_merge_params(from_node, to_node, relationship_name, properties),
                )
                if total > _WRITE_CHUNK_SIZE and (index + 1) % _WRITE_CHUNK_SIZE == 0:
                    logger.info("Merged edges %d/%d", index + 1, total)
        except Exception as e:
            logger.error(f"Failed to add edges in batch: {e}")
            raise

    async def has_edge(self, from_node: str, to_node: str, edge_label: str) -> bool:
        rows = await self._execute(
            """
            MATCH (f:Node)-[r:EDGE]->(t:Node)
            WHERE f.id = $from_id AND t.id = $to_id AND r.relationship_name = $edge_label
            RETURN COUNT(r) > 0
            """,
            {"from_id": str(from_node), "to_id": str(to_node), "edge_label": edge_label},
        )
        return bool(rows and rows[0][0])

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        if not edges:
            return []
        try:
            params: dict = {}
            clauses = []
            for i, (from_node, to_node, edge_label) in enumerate(edges):
                params[f"f_{i}"] = str(from_node)
                params[f"t_{i}"] = str(to_node)
                params[f"r_{i}"] = str(edge_label)
                clauses.append(
                    f"(f.id = $f_{i} AND t.id = $t_{i} AND r.relationship_name = $r_{i})"
                )
            rows = await self._execute(
                f"""
                MATCH (f:Node)-[r:EDGE]->(t:Node)
                WHERE {" OR ".join(clauses)}
                RETURN f.id, t.id, r.relationship_name
                """,
                params,
            )
            return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]
        except Exception as e:
            # A failed existence check is not an empty one: callers treat []
            # as "none exist" and would re-write everything. Surface errors.
            logger.error(f"Failed to check edges in batch: {e}")
            raise

    @staticmethod
    def _is_missing_table_error(error: Exception) -> bool:
        """True when the failure is just 'the table does not exist yet'.

        A missing-table error on a read genuinely means 'no rows' (empty or
        never-written graph); anything else is a real query failure that
        must not be disguised as an empty result.
        """
        return "does not exist" in str(error)

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        try:
            rows = await self._execute(
                """
                MATCH (n:Node)-[r:EDGE]-(m:Node)
                WHERE n.id = $node_id
                RETURN n.id, n.name, n.type, n.properties,
                       r.relationship_name,
                       m.id, m.name, m.type, m.properties
                """,
                {"node_id": str(node_id)},
            )
            return [
                (self._node_dict(*row[0:4]), row[4], self._node_dict(*row[5:9])) for row in rows
            ]
        except Exception as e:
            if self._is_missing_table_error(e):
                return []
            logger.error(f"Failed to get edges for node {node_id}: {e}")
            raise

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        try:
            rows = await self._execute(
                """
                MATCH (n:Node)-[r:EDGE]-(m:Node)
                WHERE n.id = $id
                RETURN DISTINCT m.id, m.name, m.type, m.properties
                """,
                {"id": str(node_id)},
            )
            return [self._node_dict(*row) for row in rows]
        except Exception as e:
            if self._is_missing_table_error(e):
                return []
            logger.error(f"Failed to get neighbours for node {node_id}: {e}")
            raise

    async def get_predecessors(
        self, node_id: Union[str, UUID], edge_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            params: dict = {"id": str(node_id)}
            label_filter = ""
            if edge_label:
                label_filter = " AND r.relationship_name = $edge_label"
                params["edge_label"] = edge_label
            rows = await self._execute(
                f"""
                MATCH (n:Node)<-[r:EDGE]-(m:Node)
                WHERE n.id = $id{label_filter}
                RETURN m.id, m.name, m.type, m.properties
                """,
                params,
            )
            return [self._node_dict(*row) for row in rows]
        except Exception as e:
            if self._is_missing_table_error(e):
                return []
            logger.error(f"Failed to get predecessors for node {node_id}: {e}")
            raise

    async def get_successors(
        self, node_id: Union[str, UUID], edge_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            params: dict = {"id": str(node_id)}
            label_filter = ""
            if edge_label:
                label_filter = " AND r.relationship_name = $edge_label"
                params["edge_label"] = edge_label
            rows = await self._execute(
                f"""
                MATCH (n:Node)-[r:EDGE]->(m:Node)
                WHERE n.id = $id{label_filter}
                RETURN m.id, m.name, m.type, m.properties
                """,
                params,
            )
            return [self._node_dict(*row) for row in rows]
        except Exception as e:
            if self._is_missing_table_error(e):
                return []
            logger.error(f"Failed to get successors for node {node_id}: {e}")
            raise

    async def get_connections(
        self, node_id: str
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        try:
            rows = await self._execute(
                """
                MATCH (n:Node)-[r:EDGE]-(m:Node)
                WHERE n.id = $node_id
                RETURN n.id, n.name, n.type, n.properties,
                       r.relationship_name, r.properties,
                       m.id, m.name, m.type, m.properties
                """,
                {"node_id": str(node_id)},
            )
            connections = []
            for row in rows:
                source = self._node_dict(*row[0:4])
                relationship = {"relationship_name": row[4]}
                relationship.update(_parse_properties_blob(row[5]))
                target = self._node_dict(*row[6:10])
                connections.append((source, relationship, target))
            return connections
        except Exception as e:
            if self._is_missing_table_error(e):
                return []
            logger.error(f"Failed to get connections for node {node_id}: {e}")
            raise

    # ------------------------------------------------------------------
    # Graph-wide operations
    # ------------------------------------------------------------------

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        try:
            node_rows = await self._execute(f"MATCH (n:Node) RETURN {self._NODE_COLUMNS}")
            formatted_nodes = [(str(row[0]), self._node_dict(*row)) for row in node_rows if row[0]]
            if not formatted_nodes:
                logger.warning("No nodes found in the database")
                return [], []

            edge_rows = await self._execute(
                "MATCH (n:Node)-[r:EDGE]->(m:Node) "
                "RETURN n.id, m.id, r.relationship_name, r.properties"
            )
            formatted_edges = [
                (
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    _parse_properties_blob(row[3]) if len(row) > 3 else {},
                )
                for row in edge_rows
                if row and len(row) >= 3
            ]

            if formatted_nodes and not formatted_edges:
                logger.debug("No edges found, creating self-referential edges for nodes")
                for node_id, _ in formatted_nodes:
                    formatted_edges.append(
                        (
                            node_id,
                            node_id,
                            "SELF",
                            {
                                "relationship_name": "SELF",
                                "relationship_type": "SELF",
                                "vector_distance": 0.0,
                            },
                        )
                    )

            logger.info(f"Retrieved {len(formatted_nodes)} nodes and {len(formatted_edges)} edges")
            return formatted_nodes, formatted_edges
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            raise

    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Return the k-hop neighborhood subgraph around seed nodes.

        Expansion is hop-by-hop instead of a single variable-length path so
        the optional ``edge_types`` filter can be pushed into each hop's
        WHERE clause (a neighbor is included iff some all-allowed path
        reaches it, matching the other backends).
        """
        if not node_ids:
            logger.warning("No node IDs provided for neighborhood retrieval.")
            return [], []

        all_ids = {str(node_id) for node_id in node_ids}
        frontier = list(all_ids)
        try:
            # depth <= 0 means no expansion: the result is the seed nodes
            # themselves (with any edges among them), matching the other
            # backends' hop semantics.
            for _ in range(max(0, int(depth))):
                if not frontier:
                    break
                params: dict = {}
                where = self._ids_where_clause("n", frontier, params)
                hop_query = f"MATCH (n:Node)-[r:EDGE]-(m:Node) WHERE {where}"
                if edge_types:
                    type_clauses = []
                    for i, edge_type in enumerate(edge_types):
                        params[f"et_{i}"] = edge_type
                        type_clauses.append(f"r.relationship_name = $et_{i}")
                    hop_query += f" AND ({' OR '.join(type_clauses)})"
                hop_query += " RETURN DISTINCT m.id"
                rows = await self._execute(hop_query, params)
                next_frontier = [row[0] for row in rows if row[0] and row[0] not in all_ids]
                all_ids.update(next_frontier)
                frontier = next_frontier

            id_list = list(all_ids)
            params = {}
            where = self._ids_where_clause("n", id_list, params)
            node_rows = await self._execute(
                f"MATCH (n:Node) WHERE {where} RETURN {self._NODE_COLUMNS}", params
            )
            formatted_nodes = [(str(row[0]), self._node_dict(*row)) for row in node_rows if row[0]]
            if not formatted_nodes:
                logger.warning("No nodes found in neighborhood.")
                return [], []

            params = {}
            src_where = self._ids_where_clause("n", id_list, params, prefix="sid")
            dst_where = self._ids_where_clause("m", id_list, params, prefix="tid")
            edge_rows = await self._execute(
                f"""
                MATCH (n:Node)-[r:EDGE]->(m:Node)
                WHERE {src_where} AND {dst_where}
                RETURN n.id, m.id, r.relationship_name, r.properties
                """,
                params,
            )
            formatted_edges = [
                (
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    _parse_properties_blob(row[3]) if len(row) > 3 else {},
                )
                for row in edge_rows
                if row and len(row) >= 3
            ]
            logger.info(
                f"Neighborhood retrieval ({depth}-hop): {len(formatted_nodes)} nodes "
                f"and {len(formatted_edges)} edges"
            )
            return formatted_nodes, formatted_edges
        except Exception as e:
            logger.error(f"Failed to get neighborhood: {e}")
            raise

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        label = node_type.__name__
        params: dict = {"label": label}
        name_clauses = []
        for i, name in enumerate(node_name or []):
            params[f"wn_{i}"] = name
            name_clauses.append(f"n.name = $wn_{i}")
        if not name_clauses:
            return [], []

        primary_rows = await self._execute(
            f"""
            MATCH (n:Node)
            WHERE n.type = $label AND ({" OR ".join(name_clauses)})
            RETURN DISTINCT n.id
            """,
            params,
        )
        primary_ids = [row[0] for row in primary_rows if row[0]]
        if not primary_ids:
            return [], []

        params = {}
        where = self._ids_where_clause("n", primary_ids, params)
        if node_name_filter_operator == "OR":
            neighbor_query = f"""
                MATCH (n:Node)-[r:EDGE]-(nbr:Node)
                WHERE {where}
                RETURN DISTINCT nbr.id
            """
        else:
            params["primary_count"] = len(primary_ids)
            neighbor_query = f"""
                MATCH (n:Node)-[r:EDGE]-(nbr:Node)
                WHERE {where}
                WITH nbr.id AS nbr_id, COUNT(DISTINCT n.id) AS matched_count
                WHERE matched_count = $primary_count
                RETURN nbr_id
            """
        nbr_rows = await self._execute(neighbor_query, params)
        neighbor_ids = [row[0] for row in nbr_rows if row[0]]

        all_ids = list({*primary_ids, *neighbor_ids})
        params = {}
        where = self._ids_where_clause("n", all_ids, params)
        node_rows = await self._execute(
            f"MATCH (n:Node) WHERE {where} RETURN {self._NODE_COLUMNS}", params
        )
        nodes = [(str(row[0]), self._node_dict(*row)) for row in node_rows if row[0]]

        params = {}
        src_where = self._ids_where_clause("a", all_ids, params, prefix="sid")
        dst_where = self._ids_where_clause("b", all_ids, params, prefix="tid")
        edge_rows = await self._execute(
            f"""
            MATCH (a:Node)-[r:EDGE]->(b:Node)
            WHERE {src_where} AND {dst_where}
            RETURN a.id, b.id, r.relationship_name, r.properties
            """,
            params,
        )
        edges = [
            (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                _parse_properties_blob(row[3]) if len(row) > 3 else {},
            )
            for row in edge_rows
            if row and len(row) >= 3
        ]
        return nodes, edges

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ):
        """Return nodes/edges filtered on Node table columns (e.g. ``type``)."""
        if not attribute_filters:
            return [], []

        where_clauses = []
        params: dict = {}
        for i, filter_dict in enumerate(attribute_filters):
            for attr, values in filter_dict.items():
                if not attr.isidentifier():
                    raise CogneeValidationError(
                        f"Invalid attribute filter key '{attr}'. Only identifiers are allowed."
                    )
                if not values:
                    continue
                value_clauses = []
                for j, value in enumerate(values):
                    param_name = f"values_{i}_{attr}_{j}"
                    params[param_name] = value
                    value_clauses.append(f"n.{attr} = ${param_name}")
                where_clauses.append("(" + " OR ".join(value_clauses) + ")")

        if not where_clauses:
            return [], []

        where_clause = " AND ".join(where_clauses)
        node_rows = await self._execute(
            f"MATCH (n:Node) WHERE {where_clause} RETURN {self._NODE_COLUMNS}", params
        )
        formatted_nodes = [(str(row[0]), self._node_dict(*row)) for row in node_rows if row[0]]
        if not formatted_nodes:
            logger.warning("No nodes found matching filters")
            return [], []

        edges_where = (
            where_clause.replace("n.", "n1.") + " AND " + where_clause.replace("n.", "n2.")
        )
        edge_rows = await self._execute(
            f"""
            MATCH (n1:Node)-[r:EDGE]->(n2:Node)
            WHERE {edges_where}
            RETURN n1.id, n2.id, r.relationship_name, r.properties
            """,
            params,
        )
        formatted_edges = [
            (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                _parse_properties_blob(row[3]) if len(row) > 3 else {},
            )
            for row in edge_rows
            if row and len(row) >= 3
        ]
        return formatted_nodes, formatted_edges

    async def delete_graph(self) -> None:
        """Clear the graph tables.

        The NeuG database is shared with the vector adapter, so only the
        Node/EDGE tables are emptied — the database files stay in place.
        """
        try:
            await self._execute("MATCH (n:Node) DETACH DELETE n")
            logger.info("Deleted all NeuG graph data (Node/EDGE tables cleared)")
        except Exception as e:
            logger.error(f"Failed to delete graph data: {e}")
            raise

    # ------------------------------------------------------------------
    # Metrics / misc
    # ------------------------------------------------------------------

    async def get_graph_metrics(self, include_optional=False) -> Dict[str, Any]:
        try:
            node_count_rows = await self._execute("MATCH (n:Node) RETURN COUNT(n)")
            edge_count_rows = await self._execute("MATCH ()-[r:EDGE]->() RETURN COUNT(r)")
            num_nodes = node_count_rows[0][0] if node_count_rows else 0
            num_edges = edge_count_rows[0][0] if edge_count_rows else 0

            edge_rows = await self._execute("MATCH (n:Node)-[r:EDGE]->(m:Node) RETURN n.id, m.id")
            edges = [(row[0], row[1]) for row in edge_rows]
            num_selfloops = sum(1 for src, dst in edges if src == dst)

            adjacency: Dict[str, set] = {}
            for src, dst in edges:
                adjacency.setdefault(src, set()).add(dst)
                adjacency.setdefault(dst, set()).add(src)

            # Connected components via BFS (undirected view).
            components: List[int] = []
            seen = set()
            node_id_rows = await self._execute("MATCH (n:Node) RETURN n.id")
            all_node_ids = [row[0] for row in node_id_rows]
            for node_id in all_node_ids:
                if node_id in seen:
                    continue
                queue = deque([node_id])
                seen.add(node_id)
                size = 0
                while queue:
                    current = queue.popleft()
                    size += 1
                    for neighbor in adjacency.get(current, ()):
                        if neighbor not in seen:
                            seen.add(neighbor)
                            queue.append(neighbor)
                components.append(size)

            mandatory_metrics = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "mean_degree": (2 * num_edges) / num_nodes if num_nodes != 0 else None,
                "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
                "num_connected_components": len(components),
                "sizes_of_connected_components": components,
            }

            if include_optional:
                # All-pairs shortest paths by BFS per node; benchmark-scale
                # graphs keep this affordable in Python.
                shortest_path_lengths = []
                for start in all_node_ids:
                    distances = {start: 0}
                    queue = deque([start])
                    while queue:
                        current = queue.popleft()
                        for neighbor in adjacency.get(current, ()):
                            if neighbor not in distances:
                                distances[neighbor] = distances[current] + 1
                                queue.append(neighbor)
                    shortest_path_lengths.extend(
                        d for node_id, d in distances.items() if node_id != start and d > 0
                    )

                triangles = 0
                for node_id in all_node_ids:
                    neighbors = adjacency.get(node_id, set())
                    for neighbor in neighbors:
                        triangles += len(neighbors & adjacency.get(neighbor, set()))
                triangles //= 6
                degree_pairs = sum(
                    degree * (degree - 1)
                    for degree in (len(adjacency.get(node_id, set())) for node_id in all_node_ids)
                )
                avg_clustering = (3 * triangles / (degree_pairs / 2)) if degree_pairs else 0.0

                optional_metrics = {
                    "num_selfloops": num_selfloops,
                    "diameter": max(shortest_path_lengths) if shortest_path_lengths else -1,
                    "avg_shortest_path_length": sum(shortest_path_lengths)
                    / len(shortest_path_lengths)
                    if shortest_path_lengths
                    else -1,
                    "avg_clustering": avg_clustering,
                }
            else:
                optional_metrics = {
                    "num_selfloops": -1,
                    "diameter": -1,
                    "avg_shortest_path_length": -1,
                    "avg_clustering": -1,
                }

            return {**mandatory_metrics, **optional_metrics}
        except Exception as e:
            logger.error(f"Failed to get graph metrics: {e}")
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "mean_degree": 0,
                "edge_density": 0,
                "num_connected_components": 0,
                "sizes_of_connected_components": [],
                "num_selfloops": -1,
                "diameter": -1,
                "avg_shortest_path_length": -1,
                "avg_clustering": -1,
            }

    async def get_disconnected_nodes(self) -> List[str]:
        """Node ids with no incident edges (NOT EXISTS subqueries are not
        available in NeuG, so this is computed from the edge list)."""
        node_rows = await self._execute("MATCH (n:Node) RETURN n.id")
        edge_rows = await self._execute("MATCH (n:Node)-[r:EDGE]-() RETURN DISTINCT n.id")
        connected = {row[0] for row in edge_rows}
        return [str(row[0]) for row in node_rows if row[0] not in connected]

    async def get_model_independent_graph_data(self) -> Dict[str, List[str]]:
        node_labels = await self._execute("MATCH (n:Node) RETURN DISTINCT labels(n)")
        rel_types = await self._execute("MATCH ()-[r:EDGE]->() RETURN DISTINCT r.relationship_name")
        return {
            "node_labels": [row[0] for row in node_labels],
            "relationship_types": [row[0] for row in rel_types],
        }

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """Retrieve a batch of (start_node, relationship_properties, end_node)."""
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")
        if limit < 0:
            raise ValueError(f"Limit must be non-negative, got {limit}")

        query = """
        MATCH (start_node:Node)-[relationship:EDGE]->(end_node:Node)
        RETURN start_node.id, start_node.name, start_node.type, start_node.properties,
               relationship.relationship_name, relationship.properties,
               end_node.id, end_node.name, end_node.type, end_node.properties
        ORDER BY start_node.id, end_node.id, relationship.relationship_name
        SKIP $offset LIMIT $limit
        """
        try:
            rows = await self._execute(query, {"offset": offset, "limit": limit})
        except Exception as e:
            logger.error(f"Failed to execute triplet query: {str(e)}")
            raise

        triplets = []
        for row in rows:
            if not row or len(row) < 10:
                continue
            relationship_properties = {"relationship_name": row[4]}
            relationship_properties.update(_parse_properties_blob(row[5]))
            triplets.append(
                {
                    "start_node": self._node_dict(*row[0:4]),
                    "relationship_properties": relationship_properties,
                    "end_node": self._node_dict(*row[6:10]),
                }
            )
        return triplets

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self):
        """Release the shared connection manager reference.

        The actual NeuG database is only closed when the reference count
        reaches zero (the vector adapter may still be using it) or at
        interpreter exit.
        """
        if not self._closed:
            self._closed = True
            self.connection_manager.release()

    async def checkpoint(self) -> None:
        # NeuG manages WAL checkpointing internally (checkpoint-on-close is
        # the default); nothing to flush explicitly.
        return None
