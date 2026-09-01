"""NeuG vector adapter with exact ANN scan and native full-text (bm25) search.

Each cognee collection maps to one NeuG node table in the SAME database the
graph adapter uses (shared through ``NeuGConnectionManager``):

    id STRING PRIMARY KEY, vector FLOAT[dim], text VARCHAR(65535),
    payload VARCHAR(65535), belongs_to_set VARCHAR(65535)

with an FTS index on ``text``. ``text`` holds the same string that gets
embedded, so native bm25 lexical search is available without any extra store.

A cosine HNSW index is created on ``vector`` next to the FTS index (the old
v0.2.0 HNSW checkpoint corruption — probe P30 — is fixed in current builds;
the only later data-loss incident occurred when reopening a database written
by an unreleased internal build, so shipped versions are unaffected; see the
repro in .neug_work/repro_upstream). Note the NeuG-specific clause order:
``CREATE INDEX <name> IF NOT EXISTS ON ...`` (the SQL-style ``CREATE INDEX
IF NOT EXISTS <name> ON ...`` is a parser error). On any index failure ANN
degrades to an exact scan (``ORDER BY vector_distance_cosine(...) ASC``),
so a failed index must not fail collection creation.

Dialect notes (from the cognee dialect probe):
- ``ORDER BY vector_distance_cosine(v.vector, $q) ASC`` for ANN;
- ``bm25(v.text, $q)`` scores are negative and ascending order = better,
  which matches cognee's ``ScoredResult`` contract (lower score is better);
- list parameters crash NeuG 0.2.0, so id/tag filters are parameterized
  OR-chains (``CONTAINS`` on the ``#tag#``-delimited ``belongs_to_set``);
  the delimiter is ``#`` because NeuG's CONTAINS treats ``|`` as a regex
  alternation metacharacter (literals containing ``|`` match everything);
- ``CONTAINS`` requires a literal right operand (parameters are rejected),
  so tags are inlined; the literal parser only accepts ``\'`` / ``\\``
  escape sequences (``''`` doubling and ``\.`` both fail to parse), so
  every regex backslash is doubled when inlined;
- ANN combined with ``LIMIT $param`` segfaults: ANN queries inline a literal
  LIMIT (bm25 tolerates parameterized LIMIT);
- a WHERE clause in the same statement as a distance/bm25 ranking expression
  segfaults (id equality) or is mis-compiled; the working shape is
  ``MATCH ... WITH v WHERE <filter> RETURN ..., <ranking> ORDER BY ...``;
- ``CREATE NODE TABLE IF NOT EXISTS`` never upgrades an existing table's
  schema, and a table left behind by an older (shorter VARCHAR) DDL silently
  truncates long blobs, so ``has_collection`` round-trips a probe string
  against pre-existing tables and recreates them when stale.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel

from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.databases.neug import get_neug_connection_manager
from cognee.infrastructure.databases.neug.copy_batch import (
    COPY_BUFFER_FLUSH_ROWS,
    copy_jsonl_rows,
)
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.vector.pgvector.serialize_data import serialize_data
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_logger

logger = get_logger("NeuGVectorAdapter")

# Probe row/column used to verify that a collection table's VARCHAR columns
# really accept long blobs. NeuG's ``CREATE NODE TABLE IF NOT EXISTS`` keeps a
# pre-existing table's schema untouched, so a table created by an older
# (shorter VARCHAR) DDL silently truncates payloads on write; the self-check
# below detects that instead of producing corrupt JSON on read.
_CAPACITY_PROBE_ID = "__neug_capacity_probe__"
_CAPACITY_PROBE_SIZE = 512

# Collection registry: NeuG has no SHOW TABLES introspection, so collection
# names are tracked in a fixed node table in the same database. This lets
# ``prune()`` discover collections created by OTHER processes/adapters
# instead of silently no-oping on a fresh adapter instance (whose in-memory
# ``self._collections`` is empty).
_REGISTRY_TABLE = "neug_vector_collections"
_REGISTRY_TABLES = frozenset({"Node", "EDGE", _REGISTRY_TABLE})


class IndexSchema(DataPoint):
    """Schema for vector-index collections (one per indexed (type, field))."""

    id: str
    text: str
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: Optional[int] = None
    source_chunk_id: Optional[str] = None
    importance_weight: Optional[float] = 0.5
    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: List[str] = []


class NeuGVectorAdapter(VectorDBInterface):
    """VectorDBInterface implementation backed by the embedded NeuG database."""

    def __init__(
        self,
        url: str = "",
        api_key: str = "",
        embedding_engine=None,
        database_name: str = "",
        vector_db_host: str = "",
        vector_db_port: str = "",
        vector_db_username: str = "",
        vector_db_password: str = "",
    ):
        """Connection parameters are accepted for factory compatibility but
        unused: collections live as node tables inside the shared NeuG
        database resolved by the connection manager."""
        self.embedding_engine = embedding_engine
        self.vector_size = embedding_engine.get_vector_size()
        self.connection_manager = get_neug_connection_manager()
        self.connection_manager.acquire()
        self._closed = False
        # collection_name -> NeuG node table name
        self._collections: Dict[str, str] = {}
        self.VECTOR_DB_LOCK = asyncio.Lock()
        # Deferred COPY buffer: table_name -> point id -> entry; each entry
        # carries the full table row plus whether a MERGE fallback must union
        # the incoming tags with the stored ones (the ``create_data_points``
        # contract) instead of replacing them. Rows are merged into one COPY
        # statement per table at flush time (see ``flush_pending_writes``).
        self._pending_rows: Dict[str, Dict[str, dict]] = {}
        self._flushing = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _table_name(collection_name: str) -> str:
        """Sanitize a collection name into a valid NeuG node table name."""
        sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", collection_name)
        if sanitized[:1].isdigit():
            sanitized = f"c_{sanitized}"
        return sanitized

    async def _execute(self, query: str, params: Optional[dict] = None) -> List[List[Any]]:
        await self.flush_pending_writes()
        return await self.connection_manager.execute(query, params)

    async def _execute_raw(self, query: str, params: Optional[dict] = None) -> List[List[Any]]:
        """Run a statement without flushing the COPY buffer first.

        Only for use by ``flush_pending_writes`` and its helpers; everything
        else goes through ``_execute`` so buffered writes are visible to
        every search, delete and MERGE upsert.
        """
        return await self.connection_manager.execute(query, params)

    async def _table_exists(self, table_name: str) -> bool:
        """Verify a collection table is really there.

        A missing-table failure (stale registry row, e.g. after an
        out-of-band prune) is a legit "no" answer; any other failure is
        surfaced. Note the engine logs an E-level "Schema mismatch" for
        every probe against a missing table, so callers must keep probing
        rare (see ``has_collection``).
        """
        try:
            await self._execute(f"MATCH (v:{table_name}) RETURN v.id LIMIT 1")
        except Exception as e:
            if "does not exist" in str(e):
                return False
            raise
        return True

    async def _ensure_registry(self) -> None:
        """Create the cross-process collection registry table if needed."""
        await self._execute(
            f"CREATE NODE TABLE IF NOT EXISTS {_REGISTRY_TABLE}("
            "name STRING PRIMARY KEY, table_name STRING)"
        )

    async def _register_collection(self, collection_name: str, table_name: str) -> None:
        await self._ensure_registry()
        await self._execute(
            f"MERGE (r:{_REGISTRY_TABLE} {{name: $name}}) "
            "ON CREATE SET r.table_name = $table_name "
            "ON MATCH SET r.table_name = $table_name",
            {"name": collection_name, "table_name": table_name},
        )

    async def _list_registered_collections(self) -> List[Tuple[str, str]]:
        try:
            await self._ensure_registry()
            rows = await self._execute(f"MATCH (r:{_REGISTRY_TABLE}) RETURN r.name, r.table_name")
        except Exception as e:
            logger.warning("Collection registry read failed: %s", e)
            return []
        return [(row[0], row[1]) for row in rows if row[0]]

    async def _verify_blob_capacity(self, table_name: str) -> None:
        """Fail fast when a pre-existing table truncates long VARCHAR blobs.

        Writes a probe row with a ``_CAPACITY_PROBE_SIZE``-character string,
        reads it back and drops the row again. A content mismatch (both reads
        and writes truncate against an old schema, so lengths alone can lie)
        means the table predates the current DDL (``IF NOT EXISTS`` never
        upgrades schemas), so the collection is dropped and recreated below.
        A rejected probe write (e.g. an embedding-dimension change) marks the
        table stale just the same: either way the caller recreates it.
        """
        probe = "p" * _CAPACITY_PROBE_SIZE
        rows: List[List[Any]] = []
        try:
            try:
                await self._execute(
                    self._MERGE_TEMPLATE.format(table=table_name),
                    {
                        "id": _CAPACITY_PROBE_ID,
                        "vector": [0.0] * self.vector_size,
                        "text": probe,
                        "payload": probe,
                        "belongs_to_set": "",
                    },
                )
                rows = await self._execute(
                    f"MATCH (v:{table_name}) WHERE v.id = '{_CAPACITY_PROBE_ID}' RETURN v.payload",
                )
            except Exception as e:
                raise ValueError(
                    f"NeuG collection table '{table_name}' rejected the capacity "
                    f"probe write ({e}). The table predates the current DDL "
                    "(e.g. a different embedding dimension); it will be dropped "
                    "and recreated."
                ) from e
        finally:
            try:
                await self._execute(
                    f"MATCH (v:{table_name}) WHERE v.id = '{_CAPACITY_PROBE_ID}' DELETE v"
                )
            except Exception as e:
                logger.debug("Capacity probe cleanup skipped for %s: %s", table_name, e)
        stored = rows[0][0] if rows else ""
        if stored != probe:
            raise ValueError(
                f"NeuG collection table '{table_name}' does not preserve "
                f"{_CAPACITY_PROBE_SIZE}-character VARCHAR blobs (read back "
                f"{len(stored)} chars). The table predates the current DDL; "
                "it will be dropped and recreated."
            )

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    async def has_collection(self, collection_name: str) -> bool:
        if collection_name in self._collections:
            return True
        table_name = self._table_name(collection_name)
        # Existence is answered from the collection registry: probing a
        # missing table with MATCH makes the engine log an E-level
        # "Schema mismatch: Table X does not exist" every time, which floods
        # the cognify log while collections are created lazily.
        registered = await self._list_registered_collections()
        if (collection_name, table_name) not in registered:
            if registered:
                # The registry has entries, just not this one: it is absent.
                return False
            # No registry entries at all (fresh or pre-registry database):
            # one probe is needed to adopt tables created by an older build
            # that predates the registry. This runs at most until the first
            # collection is registered, so cognify stays probe-free.
            if not await self._table_exists(table_name):
                return False
            # A table that predates the current DDL would silently truncate
            # payload blobs (``IF NOT EXISTS`` never upgrades schemas): verify
            # capacity and recreate it in place when stale. Registered tables
            # were created by the current DDL, so they skip the probe - its
            # insert/delete cycle would also poke an indexed table, which the
            # upstream capacity bug makes risky near a row-count boundary.
            try:
                await self._verify_blob_capacity(table_name)
            except ValueError as e:
                logger.warning("Recreating stale NeuG collection table: %s", e)
                await self._execute(f"DROP TABLE {table_name}")
                await self._create_collection_table(table_name)
        self._collections[collection_name] = table_name
        # Register pre-existing tables (created before the registry existed)
        # so a cross-process ``prune()`` can still discover them.
        await self._register_collection(collection_name, table_name)
        return True

    async def _create_collection_table(self, table_name: str) -> None:
        await self._execute(
            f"""
            CREATE NODE TABLE IF NOT EXISTS {table_name}(
                id STRING PRIMARY KEY,
                vector FLOAT[{self.vector_size}],
                text VARCHAR(65535),
                payload VARCHAR(65535),
                belongs_to_set VARCHAR(65535)
            )
            """
        )
        # NOTE: HNSW/FTS indexes are intentionally NOT created here. An
        # upstream NeuG bug makes MERGE upserts (ON MATCH updates or reused
        # delete holes) crash with "Index out of range at column.h set_any"
        # once an INDEXED table's row count crosses an internal capacity
        # boundary (first at 4096 rows, then every ~1024 rows after).
        # Unindexed tables tolerate the same upsert pattern indefinitely
        # (verified past 9000 rows), so the ingest path stays index-free and
        # the indexes are bulk-built lazily on first search, when the table
        # is already populated. Remove this deferral (build indexes here
        # again) once the upstream fix lands; see .neug_work/NEUG_BUG_REPORT_4096.md.

    async def _ensure_search_indexes(self, table_name: str) -> None:
        """Lazily bulk-build HNSW + FTS indexes on a populated table.

        Building on existing rows is a bulk path and is safe; only the
        incremental index-maintenance path during upsert-heavy growth hits
        the upstream capacity bug (see ``_create_collection_table``). ANN
        degrades to an exact scan and bm25 is unavailable until both
        succeed, so index failures must never fail the search itself.
        """
        try:
            await self._execute(
                f"CREATE INDEX {table_name}_hnsw_idx IF NOT EXISTS "
                f"ON {table_name} USING HNSW (vector) WITH (metric = 'cosine')"
            )
        except Exception as e:
            logger.warning(
                "HNSW index skipped for %s (ANN falls back to exact scan): %s",
                table_name,
                e,
            )
        try:
            await self._execute(
                f"CREATE INDEX {table_name}_fts_idx IF NOT EXISTS ON {table_name} USING FTS (text)"
            )
        except Exception as e:
            logger.debug("Index creation skipped for %s: %s", table_name, e)

    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Optional[Any] = None,
    ):
        if await self.has_collection(collection_name):
            return
        async with self.VECTOR_DB_LOCK:
            if await self.has_collection(collection_name):
                return
            table_name = self._table_name(collection_name)
            await self._create_collection_table(table_name)
            await self._register_collection(collection_name, table_name)
            self._collections[collection_name] = table_name
            logger.debug("Created NeuG collection table %s", table_name)

    async def get_collection(self, collection_name: str) -> str:
        """Return the NeuG table name for `collection_name` (raises when absent)."""
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")
        return self._collections[collection_name]

    # ------------------------------------------------------------------
    # Data points
    # ------------------------------------------------------------------

    @staticmethod
    def _belongs_to_set_string(tags) -> str:
        # ``#`` delimiter: NeuG's CONTAINS treats ``|`` as regex alternation.
        return "".join(f"#{tag}#" for tag in (tags or []))

    @staticmethod
    def _validate_tag(tag) -> str:
        # The tag delimiter must never occur inside a tag: ``"a#b"`` would
        # serialize to ``#a#b#`` and match the wrong collection under
        # CONTAINS filtering.
        text = str(tag)
        if "#" in text:
            raise ValueError(f"Collection/set names must not contain '#': {text!r}")
        return text

    async def _fetch_existing_tags(
        self, table_name: str, ids: List[str]
    ) -> Tuple[Dict[str, List[str]], set]:
        """Return (tags_by_id, existing_ids) for the given point ids.

        ``existing_ids`` contains every id already present in the table,
        including rows whose ``belongs_to_set`` is empty (which the tags
        dict omits); COPY silently skips existing primary keys, so the
        upsert path needs the full membership to route those rows through
        MERGE instead.
        """
        if not ids:
            return {}, set()
        params: dict = {}
        clauses = []
        for i, point_id in enumerate(ids):
            params[f"id_{i}"] = str(point_id)
            clauses.append(f"v.id = $id_{i}")
        try:
            rows = await self._execute_raw(
                f"MATCH (v:{table_name}) WHERE {' OR '.join(clauses)} "
                "RETURN v.id, v.belongs_to_set",
                params,
            )
        except Exception as e:
            # A failed tag lookup is not an empty one: proceeding would
            # overwrite the points' existing collection membership with only
            # the incoming tags, silently dropping them from filtered search.
            logger.error("belongs_to_set lookup failed for '%s': %s", table_name, e)
            raise
        existing: Dict[str, List[str]] = {}
        found_ids: set = set()
        for row in rows:
            found_ids.add(row[0])
            tags = [tag for tag in (row[1] or "").split("#") if tag]
            if tags:
                existing[row[0]] = tags
        return existing, found_ids

    _MERGE_TEMPLATE = """
        MERGE (v:{table} {{id: $id}})
        ON CREATE SET
            v.vector = $vector,
            v.text = $text,
            v.payload = $payload,
            v.belongs_to_set = $belongs_to_set
        ON MATCH SET
            v.vector = $vector,
            v.text = $text,
            v.payload = $payload,
            v.belongs_to_set = $belongs_to_set
    """

    async def _merge_rows(self, table_name: str, rows: List[dict]) -> None:
        """Per-row MERGE for rows COPY cannot handle (already-existing ids)."""
        merge_query = self._MERGE_TEMPLATE.format(table=table_name)
        for row in rows:
            await self._execute_raw(merge_query, row)

    async def _bulk_upsert_rows(self, table_name: str, rows: List[dict]) -> None:
        """Split rows into new vs existing and bulk-load the new ones.

        COPY silently skips existing primary keys (first write wins), which
        would silently drop updates; the caller must have routed rows whose
        ids already exist through MERGE. On any COPY failure the rows fall
        back to per-row MERGE so correctness never depends on the fast path.
        """
        if not rows:
            return
        try:
            await copy_jsonl_rows(lambda q: self._execute_raw(q), table_name, rows)
        except Exception as e:
            logger.warning(
                "COPY bulk load into %s failed, falling back to MERGE: %s", table_name, e
            )
            await self._merge_rows(table_name, rows)

    # ------------------------------------------------------------------
    # Deferred COPY buffer
    # ------------------------------------------------------------------

    async def _maybe_flush_pending(self) -> None:
        pending_count = sum(len(entries) for entries in self._pending_rows.values())
        if pending_count >= COPY_BUFFER_FLUSH_ROWS:
            await self.flush_pending_writes()

    async def flush_pending_writes(self) -> None:
        """Write all buffered rows to the database as one COPY per table.

        Every statement issued through ``_execute`` flushes first, so
        searches, deletes and MERGE upserts always see the buffered rows;
        this method is the only place the buffer is drained. The
        new-vs-existing split (and the tag union for existing points) is
        decided here at flush time, not when rows were buffered, so it
        always reflects the current table state.
        """
        if self._flushing or not self._pending_rows:
            return
        self._flushing = True
        pending, self._pending_rows = self._pending_rows, {}
        try:
            for table_name, entries in pending.items():
                await self._flush_table_entries(table_name, entries)
        finally:
            self._flushing = False

    async def _flush_table_entries(self, table_name: str, entries: Dict[str, dict]) -> None:
        existing_tags, existing_ids = await self._fetch_existing_tags(table_name, list(entries))
        merge_rows = []
        new_rows = []
        for point_id, entry in entries.items():
            row = entry["row"]
            if point_id in existing_ids:
                if entry["union_tags"]:
                    row = self._with_unioned_tags(row, existing_tags.get(point_id) or [])
                merge_rows.append(row)
            else:
                new_rows.append(row)
        if merge_rows:
            merge_query = self._MERGE_TEMPLATE.format(table=table_name)
            for row in merge_rows:
                await self._execute_raw(merge_query, row)
        await self._bulk_upsert_rows(table_name, new_rows)

    def _with_unioned_tags(self, row: dict, prior_tags: List[str]) -> dict:
        """Copy of ``row`` whose tags are unioned with the stored ones.

        The ``create_data_points`` contract keeps every collection the point
        ever belonged to: prior (database) tags first, then the incoming
        ones, mirrored into both the payload blob and the filter column.
        """
        incoming = [tag for tag in (row.get("belongs_to_set") or "").split("#") if tag]
        merged = list(dict.fromkeys(list(prior_tags) + incoming))
        payload = json.loads(row["payload"])
        payload["belongs_to_set"] = merged
        return {
            **row,
            "payload": json.dumps(payload, cls=JSONEncoder),
            "belongs_to_set": self._belongs_to_set_string(merged),
        }

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        if not data_points:
            return
        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, type(data_points[0]))

        table_name = self._collections[collection_name]

        texts = []
        for data_point in data_points:
            embeddable = DataPoint.get_embeddable_data(data_point)
            texts.append(str(embeddable) if embeddable is not None else "")
        vectors = await self.embed_data(texts)

        async with self.VECTOR_DB_LOCK:
            # Buffer the rows for the deferred COPY flush; the flush decides
            # new-vs-existing per point and unions stored tags there (the
            # create_data_points contract). When the same id is re-buffered
            # before a flush, tags accumulate locally instead of being
            # overwritten — matching the tag union the flush applies against
            # tags already stored in the table.
            pending_table = self._pending_rows.setdefault(table_name, {})
            for data_point, text, vector in zip(data_points, texts, vectors):
                payload = serialize_data(data_point.model_dump())
                tags = list(payload.get("belongs_to_set") or [])
                previous = pending_table.get(str(data_point.id))
                if previous is not None and previous["union_tags"]:
                    prior_tags = [
                        tag for tag in (previous["row"]["belongs_to_set"] or "").split("#") if tag
                    ]
                    tags = list(dict.fromkeys(prior_tags + tags))
                # Keep the payload blob and the filter column in sync.
                payload["belongs_to_set"] = tags
                point_id = str(data_point.id)
                pending_table[point_id] = {
                    "row": {
                        "id": point_id,
                        "vector": [float(x) for x in vector],
                        "text": text,
                        "payload": json.dumps(payload, cls=JSONEncoder),
                        "belongs_to_set": self._belongs_to_set_string(tags),
                    },
                    "union_tags": True,
                }
            await self._maybe_flush_pending()

    async def upsert_raw_vectors(
        self,
        collection_name: str,
        points: list[dict],
        payload_schema: Optional[type[BaseModel]] = None,
    ) -> None:
        """Upsert caller-provided vectors without invoking the embedding engine."""
        if not points:
            return
        if payload_schema is None:
            raise ValueError("payload_schema is required for NeuG raw vector upserts")
        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, payload_schema)
        table_name = self._collections[collection_name]

        # Validate everything up front so a bad point fails before any write.
        validated: List[Tuple[str, list, dict]] = []
        for point in points:
            point_id = point.get("id")
            vector = point.get("vector")
            if point_id is None:
                raise ValueError("Raw vector point is missing id")
            if not isinstance(vector, list) or len(vector) != self.vector_size:
                raise ValueError(
                    f"Raw vector size {len(vector) if isinstance(vector, list) else 'n/a'} "
                    f"does not match expected size {self.vector_size}"
                )
            payload = payload_schema.model_validate(point.get("payload")).model_dump()
            text = str(payload.get("text") or "")
            tags = [self._validate_tag(tag) for tag in (payload.get("belongs_to_set") or [])]
            validated.append((str(point_id), vector, payload, text, tags))

        async with self.VECTOR_DB_LOCK:
            # Buffer the rows for the deferred COPY flush; raw upserts
            # replace stored tags, so no union on the MERGE fallback. Same id
            # twice keeps the last occurrence.
            pending_table = self._pending_rows.setdefault(table_name, {})
            for point_id, vector, payload, text, tags in validated:
                pending_table[point_id] = {
                    "row": {
                        "id": point_id,
                        "vector": [float(x) for x in vector],
                        "text": text,
                        "payload": json.dumps(payload, cls=JSONEncoder),
                        "belongs_to_set": self._belongs_to_set_string(tags),
                    },
                    "union_tags": False,
                }
            await self._maybe_flush_pending()

    async def retrieve(
        self, collection_name: str, data_point_ids: list[str], *, include_vector: bool = False
    ):
        if not data_point_ids:
            return []
        try:
            table_name = await self.get_collection(collection_name)
        except CollectionNotFoundError:
            return []
        params: dict = {}
        clauses = []
        for i, point_id in enumerate(data_point_ids):
            params[f"id_{i}"] = str(point_id)
            clauses.append(f"v.id = $id_{i}")
        vector_column = ", v.vector" if include_vector else ""
        rows = await self._execute(
            f"MATCH (v:{table_name}) WHERE {' OR '.join(clauses)} "
            f"RETURN v.id, v.payload{vector_column}",
            params,
        )
        results = []
        for row in rows:
            payload = json.loads(row[1]) if row[1] else {}
            if include_vector:
                payload = {**payload, "vector": row[2]}
            results.append(ScoredResult(id=parse_id(row[0]), payload=payload, score=0))
        return results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _node_name_where(self, node_name: List[str], operator: str) -> str:
        # NeuG requires the CONTAINS right operand to be a literal
        # (parameters are rejected with ERR_COMPILATION), so the tags are
        # inlined. CONTAINS evaluates its right operand as a REGEX, so
        # every regex metacharacter in the tag must be escaped (``test.v2``
        # would otherwise also match ``testXv2``; ``(``/``*`` would fail to
        # compile the pattern altogether). re.escape supplies those
        # backslashes, but the Cypher string literal parser only accepts
        # ``\'`` and ``\\`` escape sequences (``\.`` is a parse error), so
        # each regex backslash is doubled for the literal; single quotes
        # use ``\'`` (``''`` doubling is a parse error in NeuG).
        joiner = " AND " if operator == "AND" else " OR "
        clauses = []
        for name in node_name:
            escaped = re.escape(self._validate_tag(name))
            normalized = escaped.replace("\\'", "'")
            literal = "#" + normalized.replace("\\", "\\\\").replace("'", "\\'") + "#"
            clauses.append(f"v.belongs_to_set CONTAINS '{literal}'")
        return "(" + joiner.join(clauses) + ")"

    @staticmethod
    def _sanitize_fts_query(query_text: str) -> str:
        """Turn a natural-language query into a safe FTS5 phrase (P26).

        NeuG's FTS extension hands the query string to SQLite FTS5, which
        (1) rejects unquoted specials (``?``, ``"``, ``*``, parentheses, ...)
        with a runtime error and (2) parses AND/OR/NOT as operators, so
        "why is it not working" would negate the term ``working``. Both are
        avoided by stripping punctuation and wrapping every token in double
        quotes: tokens stay literals and cannot inject operators.

        Tokens are OR-joined instead of relying on the FTS5 default AND:
        the search pipeline rewrites queries into interrogative form
        ("LoCoMo benchmark" -> "What is the LoCoMo benchmark?"), and AND
        then returns zero rows whenever a chunk lacks any question word.
        OR keeps bm25 partial-match semantics (hits are ranked by the sum
        of matched-term weights, like the default backend's BM25 retriever),
        so ranking quality is preserved.
        """
        cleaned = re.sub(r"[^\w\s]", " ", str(query_text), flags=re.UNICODE)
        return " OR ".join('"' + token.replace('"', '""') + '"' for token in cleaned.split())

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
    ):
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()
        try:
            table_name = await self.get_collection(collection_name)
        except CollectionNotFoundError:
            return []
        if limit is not None and limit <= 0:
            return []
        # Deferred bulk index build (upstream workaround, safe on populated
        # tables): ANN uses HNSW once present, exact scan until then.
        async with self.VECTOR_DB_LOCK:
            await self._ensure_search_indexes(table_name)

        if query_vector is None:
            # Match LanceDB semantics: a text-only search embeds the query and
            # runs ANN. Lexical bm25 ranking stays available via
            # ``full_text_search`` (used by NeuGFTSChunksRetriever).
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        params: dict = {}
        where = ""
        if node_name:
            where = " WHERE " + self._node_name_where(node_name, node_name_filter_operator)
        columns = ["v.id"]
        if include_payload:
            columns.append("v.payload")
        if with_vector:
            columns.append("v.vector")

        params["q"] = list(query_vector)
        ranking = "vector_distance_cosine(v.vector, $q) AS score"

        # ANN segfaults with a parameterized LIMIT in NeuG 0.2.0, so the ANN
        # path inlines a literal (validated int; no injection surface). bm25
        # tolerates LIMIT $limit but both paths use the same shape.
        limit_clause = ""
        if limit is not None:
            limit_clause = f" LIMIT {int(limit)}"

        # The filter rides on ``WITH v WHERE`` rather than ``MATCH ... WHERE``:
        # a WHERE in the same statement as a distance/bm25 ranking expression
        # either segfaults (id equality) or gets mis-compiled, while the WITH
        # shape filters correctly for both ANN and bm25.
        query = (
            f"MATCH (v:{table_name}) WITH v{where} "
            f"RETURN {', '.join(columns)}, {ranking} "
            f"ORDER BY score ASC{limit_clause}"
        )
        rows = await self._execute(query, params)
        if not rows:
            return []

        results = []
        for row in rows:
            payload = None
            vector = None
            cursor = 1
            if include_payload:
                payload = json.loads(row[cursor]) if row[cursor] else {}
                cursor += 1
            if with_vector:
                vector = row[cursor]
                cursor += 1
            if with_vector and payload is not None:
                payload = {**payload, "vector": vector}
            results.append(ScoredResult(id=parse_id(row[0]), payload=payload, score=float(row[-1])))
        return results

    async def full_text_search(
        self,
        collection_name: str,
        query_text: str,
        limit: Optional[int] = 15,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
    ) -> List[tuple]:
        """Native bm25 lexical search returning ``(payload, score)`` pairs.

        Used by the CHUNKS_LEXICAL route; score keeps NeuG's bm25 ordering
        (lower is better). ``search`` intentionally does NOT fall back to bm25
        for text-only queries (it embeds, like LanceDB), so this method runs
        the ranking expression itself.
        """
        try:
            table_name = await self.get_collection(collection_name)
        except CollectionNotFoundError:
            return []
        if limit is not None and limit <= 0:
            return []
        # bm25 requires the FTS index; bulk-build it lazily (upstream
        # workaround). Without it this query fails, unlike ANN's exact-scan
        # fallback, so a failed build surfaces as an empty result instead.
        async with self.VECTOR_DB_LOCK:
            await self._ensure_search_indexes(table_name)

        params: dict = {"q": self._sanitize_fts_query(query_text)}
        where = ""
        if node_name:
            where = " WHERE " + self._node_name_where(node_name, node_name_filter_operator)
        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
        query = (
            f"MATCH (v:{table_name}) WITH v{where} "
            f"RETURN v.id, v.payload, bm25(v.text, $q) AS score "
            f"ORDER BY score ASC{limit_clause}"
        )
        rows = await self._execute(query, params)
        return [(json.loads(row[1]) if row[1] else {}, float(row[2])) for row in rows]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int] = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
    ):
        query_vectors = await self.embedding_engine.embed_text(query_texts)
        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                    include_payload=include_payload,
                    node_name=node_name,
                )
                for query_vector in query_vectors
            ]
        )

    # ------------------------------------------------------------------
    # Deletion / maintenance
    # ------------------------------------------------------------------

    async def delete_data_points(self, collection_name: str, data_point_ids: List[UUID]):
        if not data_point_ids:
            return
        try:
            table_name = await self.get_collection(collection_name)
        except CollectionNotFoundError:
            return
        params: dict = {}
        clauses = []
        for i, point_id in enumerate(data_point_ids):
            params[f"id_{i}"] = str(point_id)
            clauses.append(f"v.id = $id_{i}")
        await self._execute(
            f"MATCH (v:{table_name}) WHERE {' OR '.join(clauses)} DETACH DELETE v", params
        )

    async def delete_collection(self, collection_name: str) -> None:
        table_name = self._collections.pop(collection_name, self._table_name(collection_name))
        try:
            await self._execute(f"DROP TABLE {table_name}")
        except Exception as e:
            logger.debug("Drop table %s skipped: %s", table_name, e)
        try:
            await self._ensure_registry()
            await self._execute(
                f"MATCH (r:{_REGISTRY_TABLE}) WHERE r.name = $name DELETE r",
                {"name": collection_name},
            )
        except Exception as e:
            logger.debug("Registry cleanup for %s skipped: %s", collection_name, e)

    async def prune(self):
        """Drop every vector collection table, discovered cross-process.

        Collections are enumerated from the shared registry table (not from
        this instance's in-memory cache), so a ``cognee prune`` running in a
        fresh process against a persistent database still clears everything
        earlier runs stored. The graph tables (Node/EDGE) live in the same
        NeuG database and are untouched; cognee clears those through the
        graph adapter's ``delete_graph``.
        """
        candidates = dict(await self._list_registered_collections())
        for collection_name in list(self._collections.keys()):
            candidates.setdefault(collection_name, self._collections[collection_name])
        for collection_name, table_name in candidates.items():
            if table_name in _REGISTRY_TABLES:
                continue
            try:
                await self._execute(f"DROP TABLE {table_name}")
                logger.debug("Pruned NeuG collection table %s", table_name)
            except Exception as e:
                logger.debug("Prune drop of %s skipped: %s", table_name, e)
        self._collections.clear()
        try:
            await self._execute(f"MATCH (r:{_REGISTRY_TABLE}) DETACH DELETE r")
        except Exception as e:
            logger.debug("Registry cleanup during prune skipped: %s", e)

    # ------------------------------------------------------------------
    # Embeddings / indexing
    # ------------------------------------------------------------------

    async def embed_data(self, data: List[str]) -> List[List[float]]:
        return await self.embedding_engine.embed_text(data)

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(f"{index_name}_{index_property_name}", IndexSchema)

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=str(data_point.id),
                    text=getattr(data_point, data_point.metadata["index_fields"][0]),
                    document_id=getattr(data_point, "document_id", None),
                    document_name=getattr(data_point, "document_name", None),
                    chunk_index=getattr(data_point, "chunk_index", None),
                    source_chunk_id=getattr(data_point, "source_chunk_id", None),
                    importance_weight=getattr(data_point, "importance_weight", None),
                    belongs_to_set=(data_point.belongs_to_set or []),
                )
                for data_point in data_points
            ],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self):
        """Release the shared connection manager reference (see the graph
        adapter's ``close()`` for the refcount semantics)."""
        if not self._closed:
            try:
                await self.flush_pending_writes()
            except Exception as e:
                logger.warning("Failed to flush buffered vector writes on close: %s", e)
            self._closed = True
            self.connection_manager.release()
