from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.lexical_retriever import LexicalRetriever
from cognee.shared.logging_utils import get_logger

logger = get_logger("NeuGFTSChunksRetriever")


class NeuGFTSChunksRetriever(LexicalRetriever):
    """CHUNKS_LEXICAL retriever backed by NeuG's native full-text (bm25) index.

    The default ``BM25ChunksRetriever`` loads every DocumentChunk out of the
    graph engine and scores them with an in-memory Okapi BM25. When the vector
    backend is NeuG, the chunk text already lives in a node table with an FTS
    index on ``text``, so lexical ranking can run inside the database via
    ``bm25(...)`` instead of pulling the whole corpus into Python.

    ``get_context_from_objects`` and ``get_completion_from_context`` are reused
    unchanged from ``LexicalRetriever`` — the payload shape (a dict carrying
    ``id`` and ``text``) is identical, so downstream consumers behave the same.
    """

    # Collection layout mirrors ``index_data_points``: one collection per
    # indexed (type, field) named ``{type_name}_{field_name}``.
    COLLECTION_NAME = "DocumentChunk_text"

    def __init__(self, top_k: int = 15, with_scores: bool = False, session_id=None):
        # tokenizer/scorer are unused: ranking happens inside NeuG's FTS index,
        # not in the in-memory model the parent was designed around. Passing
        # no-op callables satisfies the parent's constructor contract while the
        # overridden initialize()/get_retrieved_objects() bypass that path.
        super().__init__(
            tokenizer=lambda text: [],
            scorer=lambda query_tokens, chunk_tokens: 0.0,
            top_k=top_k,
            with_scores=with_scores,
            session_id=session_id,
        )

    async def initialize(self):
        """Verify the DocumentChunk FTS collection exists so empty corpora fail
        fast with ``NoDataError``, matching the in-memory retriever's contract."""
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing NeuGFTSChunksRetriever against NeuG FTS index")
            vector_engine = await get_vector_engine_async()
            try:
                has_collection = await vector_engine.has_collection(self.COLLECTION_NAME)
            except Exception as e:
                logger.error("NeuG vector engine initialization failed")
                raise NoDataError("NeuG vector engine initialization failed") from e

            if not has_collection:
                logger.error("No DocumentChunk FTS collection found in NeuG.")
                raise NoDataError("No DocumentChunk FTS collection found in NeuG.")

            self._initialized = True

    async def get_retrieved_objects(self, query: str) -> list:
        """Rank DocumentChunks for ``query`` using NeuG's native bm25 index."""
        if not self._initialized:
            await self.initialize()

        if not query:
            logger.warning("Empty query for NeuG FTS retrieval")
            return []

        vector_engine = await get_vector_engine_async()
        try:
            scored_payloads = await vector_engine.full_text_search(
                self.COLLECTION_NAME, query, limit=self.top_k
            )
        except Exception as e:
            logger.error("NeuG full-text search failed: %s", str(e))
            return []

        # NeuG's bm25 score is lower-is-better (more negative = stronger match),
        # which is the opposite of the in-memory BM25's higher-is-better
        # convention. Negate so the outward-facing contract (larger = better)
        # stays consistent for consumers that compare scores.
        results = [(payload, -score) for payload, score in scored_payloads]

        logger.info(
            "Retrieved %d chunks from NeuG FTS for query (len=%d)",
            len(results),
            len(query),
        )

        if self.with_scores:
            return results
        return [payload for payload, _ in results]
