"""NeuG backend adapter smoke tests (graph + vector + shared database).

These run against the embedded NeuG database through the shared connection
manager. They require the ``neug`` Python bindings, which are not a cognee
dependency, so the whole module is skipped when ``neug`` is not importable.

Every test gets a fresh temporary database (via ``NEUG_DB_PATH``) and resets
the process-level connection-manager singleton so tests never share state.
"""

from __future__ import annotations

import uuid

import pytest

neug = pytest.importorskip("neug")

import cognee.infrastructure.databases.neug.connection_manager as _cm  # noqa: E402
from cognee.infrastructure.databases.graph.neug.adapter import (  # noqa: E402
    NeuGGraphAdapter,
)
from cognee.infrastructure.databases.vector.neug.NeuGVectorAdapter import (  # noqa: E402
    NeuGVectorAdapter,
)
from cognee.infrastructure.engine import DataPoint  # noqa: E402

VECTOR_DIM = 8


@pytest.fixture()
def neug_db(tmp_path, monkeypatch):
    """Point the shared NeuG connection manager at a fresh temporary database."""
    if _cm._manager is not None:
        _cm._manager.shutdown()
    _cm._manager = None
    monkeypatch.setenv("NEUG_DB_PATH", str(tmp_path / "neug_db"))
    yield tmp_path / "neug_db"
    if _cm._manager is not None:
        _cm._manager.shutdown()
    _cm._manager = None


def _vec(text: str) -> list:
    """Deterministic pseudo-embedding so tests never call a real engine."""
    digest = (text * 8).encode()
    return [digest[i] / 255.0 for i in range(VECTOR_DIM)]


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return VECTOR_DIM

    def get_batch_size(self):
        return 8

    async def embed_text(self, texts):
        return [_vec(t) for t in texts]


class _Chunk(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


def _chunk(text: str, tags=None) -> _Chunk:
    return _Chunk(
        id=uuid.uuid5(uuid.NAMESPACE_URL, text),
        text=text,
        belongs_to_set=tags or [],
    )


class _Node:
    """Minimal node stand-in exposing the model_dump() contract."""

    def __init__(self, id, name, type, **extra):
        self.id = id
        self.name = name
        self.type = type
        self.extra = extra

    def model_dump(self):
        return {"id": self.id, "name": self.name, "type": self.type, **self.extra}


# ---------------------------------------------------------------------------
# Graph adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_crud_and_traversal(neug_db):
    g = NeuGGraphAdapter()
    try:
        await g.add_nodes(
            [
                _Node("n1", "Alice", "Person", age=30),
                _Node("n2", "Bob", "Person", age=25),
                _Node("n3", "Acme", "Entity"),
            ]
        )
        await g.add_edges([("n1", "n2", "knows", {"since": 2020}), ("n2", "n3", "works_at", {})])

        assert await g.is_empty() is False
        assert (await g.get_node("n1"))["name"] == "Alice"
        assert sorted(n["name"] for n in await g.get_nodes(["n1", "n3"])) == ["Acme", "Alice"]
        assert await g.has_edge("n1", "n2", "knows") is True
        assert await g.has_edge("n1", "n3", "knows") is False
        # Contract: return the subset of the input triples that exist.
        assert await g.has_edges([("n1", "n2", "knows"), ("n1", "n3", "knows")]) == [
            ("n1", "n2", "knows")
        ]
        assert [n["id"] for n in await g.get_neighbors("n1")] == ["n2"]

        nodes, edges = await g.get_graph_data()
        assert len(nodes) == 3 and len(edges) == 2

        nodes, edges = await g.get_neighborhood(["n1"], depth=2)
        assert sorted(i for i, _ in nodes) == ["n1", "n2", "n3"]

        nodes, _ = await g.get_filtered_graph_data([{"type": ["Person"]}])
        assert sorted(i for i, _ in nodes) == ["n1", "n2"]

        metrics = await g.get_graph_metrics()
        assert metrics["num_nodes"] == 3 and metrics["num_edges"] == 2
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_graph_raw_cypher_and_introspection_shim(neug_db):
    g = NeuGGraphAdapter()
    try:
        await g.add_nodes([_Node("n1", "Alice", "Person"), _Node("n2", "Bob", "Person")])
        await g.add_edge("n1", "n2", "knows")

        rows = await g.query("MATCH (n:Node) WHERE n.type = 'Person' RETURN n.name ORDER BY n.name")
        # Single-column Cypher rows come back unwrapped as scalars.
        assert rows == ["Alice", "Bob"]

        # type() is not a NeuG function; the adapter rewrites it to label().
        rel_types = await g.query("MATCH ()-[r:EDGE]->() RETURN DISTINCT type(r)")
        assert {"EDGE"} == set(rel_types)

        # The schema-introspection shape the NaturalLanguageRetriever relies on.
        introspection = await g.query(
            "MATCH (n) UNWIND keys(n) AS prop RETURN DISTINCT labels(n) AS NodeLabels, "
            "collect(DISTINCT prop) AS Properties"
        )
        assert introspection, "introspection shim returned nothing"

        # Unknown labels (LLM-generated multi-label Cypher) fall back onto the
        # single-table schema instead of erroring out.
        fallback = await g.query("MATCH (e:Person) RETURN e.name ORDER BY e.name")
        assert fallback == ["Alice", "Bob"]
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_graph_delete_paths(neug_db):
    g = NeuGGraphAdapter()
    try:
        await g.add_nodes([_Node("a", "A", "T"), _Node("b", "B", "T")])
        await g.add_edge("a", "b", "rel")
        await g.delete_node("b")
        assert await g.get_node("b") is None
        await g.delete_nodes(["a"])
        assert await g.is_empty() is True

        await g.add_nodes([_Node("x", "X", "T")])
        await g.delete_graph()
        assert await g.is_empty() is True
    finally:
        await g.close()


# ---------------------------------------------------------------------------
# Vector adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_collection_and_ann_search(neug_db):
    v = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    try:
        assert await v.has_collection("DataPoint_text") is False
        await v.create_collection("DataPoint_text")
        assert await v.has_collection("DataPoint_text") is True

        points = [
            _chunk("the cat sat on the mat", ["set_a"]),
            _chunk("dogs bark at strangers", ["set_b"]),
            _chunk("cats and dogs are pets", ["set_a", "set_b"]),
        ]
        await v.create_data_points("DataPoint_text", points)

        results = await v.search(
            "DataPoint_text",
            query_vector=_vec("the cat sat on the mat"),
            limit=3,
            include_payload=True,
        )
        assert len(results) == 3
        assert results[0].payload["text"] == "the cat sat on the mat"
        scores = [r.score for r in results]
        assert scores == sorted(scores), "ANN results must be lower-is-better ordered"

        # Text-only search embeds the query (LanceDB semantics, e.g. the
        # SUMMARIES retriever relies on it): identical text embeds to the
        # identical vector and must rank first.
        text_hits = await v.search(
            "DataPoint_text",
            query_text="dogs bark at strangers",
            limit=1,
            include_payload=True,
        )
        assert text_hits[0].payload["text"] == "dogs bark at strangers"
    finally:
        await v.close()


@pytest.mark.asyncio
async def test_vector_bm25_and_tag_filters(neug_db):
    v = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    try:
        points = [
            _chunk("the cat sat on the mat", ["set_a"]),
            _chunk("dogs bark at strangers", ["set_b"]),
            _chunk("cats and dogs are pets", ["set_a", "set_b"]),
        ]
        await v.create_data_points("DataPoint_text", points)

        fts = await v.full_text_search("DataPoint_text", "dogs bark", limit=3)
        assert fts[0][0]["text"] == "dogs bark at strangers"

        # P24: bm25 matches exact tokens (no stemming), and the OR tag filter
        # drops bm25 hits outside node_name. "dogs" matches two chunks but
        # "dogs bark at strangers" (set_b only) is filtered out by set_a.
        or_filtered = await v.full_text_search(
            "DataPoint_text", "dogs", limit=5, node_name=["set_a"]
        )
        assert {payload["text"] for payload, _ in or_filtered} == {"cats and dogs are pets"}

        and_filtered = await v.full_text_search(
            "DataPoint_text",
            "pets",
            limit=5,
            node_name=["set_a", "set_b"],
            node_name_filter_operator="AND",
        )
        assert [payload["text"] for payload, _ in and_filtered] == ["cats and dogs are pets"]
    finally:
        await v.close()


@pytest.mark.asyncio
async def test_vector_belongs_to_set_merge(neug_db):
    v = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    try:
        point = _chunk("shared chunk text", ["set_a"])
        await v.create_data_points("DataPoint_text", [point])
        # Re-upsert the same id with a different tag set: tags must accumulate.
        await v.create_data_points("DataPoint_text", [_chunk("shared chunk text", ["set_b"])])

        retrieved = await v.retrieve("DataPoint_text", [str(point.id)])
        merged = retrieved[0].payload.get("belongs_to_set") or []
        assert set(merged) == {"set_a", "set_b"}
    finally:
        await v.close()


@pytest.mark.asyncio
async def test_vector_retrieve_and_delete(neug_db):
    v = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    try:
        points = [_chunk("alpha beta"), _chunk("gamma delta")]
        await v.create_data_points("DataPoint_text", points)

        retrieved = await v.retrieve("DataPoint_text", [str(points[0].id)])
        assert len(retrieved) == 1 and retrieved[0].payload["text"] == "alpha beta"

        await v.delete_data_points("DataPoint_text", [str(points[0].id)])
        assert await v.retrieve("DataPoint_text", [str(points[0].id)]) == []

        await v.delete_collection("DataPoint_text")
        assert await v.has_collection("DataPoint_text") is False
    finally:
        await v.close()


@pytest.mark.asyncio
async def test_vector_long_payload_roundtrip(neug_db):
    """Payload blobs longer than any short VARCHAR default must survive intact."""
    v = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    try:
        long_text = "long haul storage fidelity probe " * 40  # > 1000 chars
        point = _chunk(long_text, ["set_a"])
        await v.create_data_points("DataPoint_text", [point])

        retrieved = await v.retrieve("DataPoint_text", [str(point.id)])
        assert retrieved[0].payload["text"] == long_text
    finally:
        await v.close()


@pytest.mark.asyncio
async def test_vector_payload_survives_database_reopen(neug_db):
    """Data must survive a full database close/reopen cycle (checkpoint + WAL).

    Regression guard for the NeuG 0.2.0 HNSW/checkpoint bug (dialect probe
    P30): a broken close-time checkpoint makes the WAL replay truncate every
    VARCHAR to 255 characters. The adapter creates no HNSW index, so closing
    and reopening the database must leave long payloads intact.
    """
    long_text = "survives the checkpoint cycle " * 40  # > 1000 chars
    point = _chunk(long_text, ["set_a"])

    v = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    await v.create_data_points("DataPoint_text", [point])
    await v.close()  # refcount -> 0: the shared database is really closed

    v2 = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    try:
        retrieved = await v2.retrieve("DataPoint_text", [str(point.id)])
        assert retrieved[0].payload["text"] == long_text
    finally:
        await v2.close()


@pytest.mark.asyncio
async def test_vector_stale_table_is_recreated(neug_db):
    """A collection table left behind by an older (short VARCHAR) DDL must be
    detected and recreated instead of silently truncating payloads: NeuG's
    ``CREATE NODE TABLE IF NOT EXISTS`` never upgrades an existing schema."""
    from cognee.infrastructure.databases.neug import get_neug_connection_manager

    cm = get_neug_connection_manager()
    cm.acquire()
    await cm.execute(
        "CREATE NODE TABLE DataPoint_text("
        "id STRING PRIMARY KEY, vector FLOAT[8], text VARCHAR(255), "
        "payload VARCHAR(255), belongs_to_set VARCHAR(255))"
    )
    cm.release()

    v = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    try:
        long_text = "stale schema recreation probe " * 30  # ~900 chars
        point = _chunk(long_text)
        await v.create_data_points("DataPoint_text", [point])

        retrieved = await v.retrieve("DataPoint_text", [str(point.id)])
        assert retrieved[0].payload["text"] == long_text
    finally:
        await v.close()


# ---------------------------------------------------------------------------
# Shared database: graph + vector adapters coexist on one NeuG DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_database_coexistence(neug_db):
    v = NeuGVectorAdapter(embedding_engine=_FakeEmbeddingEngine())
    g = NeuGGraphAdapter()
    try:
        await v.create_data_points("DataPoint_text", [_chunk("the cat sat on the mat")])
        await g.add_nodes([_Node("n1", "Alice", "Person")])

        # Both channels serve reads while the other adapter holds a reference.
        assert (await g.get_node("n1"))["name"] == "Alice"
        hits = await v.full_text_search("DataPoint_text", "cat", limit=5)
        assert hits and hits[0][0]["text"] == "the cat sat on the mat"

        # Closing the graph adapter must not tear the shared connection down.
        await g.close()
        still = await v.full_text_search("DataPoint_text", "cat", limit=5)
        assert still, "vector search broke after graph adapter close"

        # Vector prune drops collection tables but must not touch Node/EDGE.
        await v.prune()
        assert await v.has_collection("DataPoint_text") is False
    finally:
        await v.close()

    # A fresh graph handle confirms the graph tables survived the vector prune.
    g2 = NeuGGraphAdapter()
    try:
        assert await g2.is_empty() is False, "vector prune wiped graph tables"
        assert (await g2.get_node("n1"))["name"] == "Alice"
    finally:
        await g2.close()
