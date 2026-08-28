import json
from typing import Any, Optional
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import INTERNAL_PROPERTY, is_internal_node
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions import SearchTypeNotSupported
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

logger = get_logger("NaturalLanguageRetriever")


class GuidanceSchemaRow(str):
    """Sentinel for free-form schema guidance rows in introspection results.

    Single-table backends append guidance text (e.g. how to express cognee
    semantic types on their fixed schema) alongside the real schema rows.
    A dedicated type keeps guidance distinguishable from a real label that
    happens to carry a single property key, which a length-based heuristic
    cannot tell apart.
    """


# Default node-schema block for the Cypher-generation prompt, used when the
# backend cannot answer the schema-introspection query (or returns nothing).
# It describes cognee's per-type labels, as stored on label-per-type backends.
_DEFAULT_NODE_SCHEMAS = """\
- EntityType
Properties: description, ontology_valid, name, created_at, type, version, topological_rank, updated_at, metadata, id
Purpose: Represents the categories or classifications for entities in the database.

- Entity
Properties: description, ontology_valid, name, created_at, type, version, topological_rank, updated_at, metadata, id
Purpose: Represents individual entities that belong to a specific type or classification.

- TextDocument
Properties: raw_data_location, name, mime_type, external_metadata, created_at, type, version, topological_rank, updated_at, metadata, id
Purpose: Represents documents containing text data, along with metadata about their storage and format.

- DocumentChunk
Properties: version, created_at, type, topological_rank, cut_type, text, metadata, chunk_index, chunk_size, updated_at, id
Purpose: Represents segmented portions of larger documents, useful for processing or analysis at a more granular level.

- TextSummary
Properties: topological_rank, metadata, id, type, updated_at, created_at, text, version
Purpose: Represents summarized content generated from larger text documents, retaining essential information and metadata."""


def _schema_row_from_dict(row: dict):
    """Extract ``(labels, properties)`` from a dict-shaped introspection row.

    The Neo4j Cypher path returns ``result.data()`` rows as dicts keyed by
    the RETURN aliases (``NodeLabels``/``Properties``); without this, those
    rows fall through to ``str(row)`` and the prompt ends up with Python
    dict reprs instead of a readable schema.
    """
    labels = None
    properties = None
    for key, value in row.items():
        if key.lower() in ("nodelabels", "labels", "label") and labels is None:
            labels = value
        elif key.lower() in ("properties", "props", "propertykeys") and properties is None:
            properties = value
    if labels is None:
        list_values = [value for value in row.values() if isinstance(value, (list, tuple))]
        if len(list_values) >= 2:
            labels, properties = list_values[0], list_values[1]
        elif len(list_values) == 1:
            labels = list_values[0]
    return labels, properties


def _format_node_schemas(node_schemas) -> str:
    """Render introspection rows into the prompt's node-schema section.

    Rows come back as ``(labels, property keys)`` pairs from the
    ``RETURN DISTINCT labels(n), collect(DISTINCT prop)`` introspection
    query, as dicts on backends whose Cypher pass-through returns
    ``result.data()`` rows (Neo4j), or as ``GuidanceSchemaRow`` sentinels
    for free-form guidance. Falls back to the hardcoded per-type schema
    when the backend returned nothing usable.
    """
    if not node_schemas:
        return _DEFAULT_NODE_SCHEMAS
    rendered = []
    for row in node_schemas:
        if isinstance(row, GuidanceSchemaRow):
            rendered.append(f"- {row}")
            continue
        if isinstance(row, dict):
            row = _schema_row_from_dict(row)
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            rendered.append(str(row))
            continue
        labels, properties = row
        label_text = (
            ", ".join(str(label) for label in labels)
            if isinstance(labels, (list, tuple))
            else str(labels)
        )
        props_text = (
            ", ".join(str(prop) for prop in properties)
            if isinstance(properties, (list, tuple))
            else str(properties)
        )
        rendered.append(f"- {label_text}\nProperties: {props_text}")
    return "\n\n".join(rendered) if rendered else _DEFAULT_NODE_SCHEMAS


def _is_internal_schema_row(row: Any) -> bool:
    """True for a node-schema row that describes internal nodes.

    On backends with per-node property keys (Neo4j), internal nodes are the only
    ones carrying the ``is_internal`` key, so a schema row whose collected
    property list contains it describes internal nodes. On backends with a fixed
    column set (Ladybug/Kuzu) the marker lives inside the serialized
    ``properties`` blob and never shows up as a key, so this is a no-op there —
    the fixed columns carry nothing internal-specific to hide.
    """
    if isinstance(row, dict):
        values = row.values()
    elif isinstance(row, (list, tuple)):
        values = row
    else:
        return False
    for value in values:
        if value == INTERNAL_PROPERTY:
            return True
        if isinstance(value, (list, tuple, set)) and INTERNAL_PROPERTY in value:
            return True
    return False


def _contains_internal_node(value: Any) -> bool:
    """True when a raw query result value contains an internal graph node.

    Handles deserialized node dicts (``is_internal`` at the top level), nested
    containers, and Ladybug/Kuzu raw rows where node properties come back as a
    serialized JSON string column.
    """
    if isinstance(value, dict):
        if is_internal_node(value):
            return True
        return any(_contains_internal_node(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_internal_node(item) for item in value)
    if isinstance(value, str) and INTERNAL_PROPERTY in value:
        try:
            parsed = json.loads(value)
        except ValueError:
            return False
        return isinstance(parsed, dict) and is_internal_node(parsed)
    return False


class NaturalLanguageRetriever(BaseRetriever):
    """
    Retriever for handling natural language search.

    Public methods include:

    - get_context: Retrieves relevant context using a natural language query converted to
    Cypher.
    - get_completion: Returns a completion based on the query and context.
    """

    def __init__(
        self,
        system_prompt_path: str = "natural_language_retriever_system.txt",
        max_attempts: int = 3,
        session_id: Optional[str] = None,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.system_prompt_path = system_prompt_path
        self.max_attempts = max_attempts
        self.session_id = session_id

    async def _get_graph_schema(self, graph_engine) -> tuple:
        """Retrieve the node and edge schemas from the graph database."""
        node_schemas = await graph_engine.query(
            """
            MATCH (n)
            UNWIND keys(n) AS prop
            RETURN DISTINCT labels(n) AS NodeLabels, collect(DISTINCT prop) AS Properties;
            """
        )
        edge_schemas = await graph_engine.query(
            """
            MATCH ()-[r]->()
            UNWIND keys(r) AS key
            RETURN DISTINCT key;
            """
        )
        # Internal nodes (e.g. per-user preference state) must never be surfaced,
        # so keep their labels out of the schema handed to the LLM.
        node_schemas = [row for row in node_schemas or [] if not _is_internal_schema_row(row)]
        return node_schemas, edge_schemas

    async def _generate_cypher_query(
        self, query: str, node_schemas, edge_schemas, previous_attempts=None
    ) -> str:
        """Generate a Cypher query using LLM based on natural language query and schema information."""
        system_prompt = render_prompt(
            self.system_prompt_path,
            context={
                "node_schemas": _format_node_schemas(node_schemas),
                "edge_schemas": edge_schemas,
                "previous_attempts": previous_attempts or "No attempts yet",
            },
        )

        return await LLMGateway.acreate_structured_output(
            text_input=query,
            system_prompt=system_prompt,
            response_model=str,
        )

    async def _execute_cypher_query(self, query: str, graph_engine: GraphDBInterface) -> Any:
        """Execute the natural language query against Neo4j with multiple attempts."""
        node_schemas, edge_schemas = await self._get_graph_schema(graph_engine)
        previous_attempts = ""
        cypher_query = ""

        for attempt in range(self.max_attempts):
            logger.info(f"Starting attempt {attempt + 1}/{self.max_attempts} for query generation")
            try:
                cypher_query = await self._generate_cypher_query(
                    query, node_schemas, edge_schemas, previous_attempts
                )

                logger.info(
                    f"Executing generated Cypher query (attempt {attempt + 1}): {cypher_query[:100]}..."
                    if len(cypher_query) > 100
                    else cypher_query
                )
                context = await graph_engine.query(cypher_query)

                if isinstance(context, list):
                    # Internal nodes (e.g. per-user preference state) must never
                    # reach a user; drop any row that contains one.
                    context = [row for row in context if not _contains_internal_node(row)]

                if context:
                    result_count = len(context) if isinstance(context, list) else 1
                    logger.info(
                        f"Successfully executed query (attempt {attempt + 1}): returned {result_count} result(s)"
                    )
                    return context

                previous_attempts += f"Query: {cypher_query} -> Result: None\n"

            except Exception as e:
                previous_attempts += f"Query: {cypher_query if 'cypher_query' in locals() else 'Not generated'} -> Executed with error: {e}\n"
                logger.error(f"Error executing query: {str(e)}")

        logger.warning(
            f"Failed to get results after {self.max_attempts} attempts for query: '{query[:50]}...'"
        )
        return []

    async def get_retrieved_objects(self, query: str) -> Any:
        graph_engine = await get_graph_engine()

        # Cypher support is declared on the adapter class
        # (GraphDBInterface.supports_cypher_queries), so backends like Postgres
        # and Turso are excluded without importing optional backend packages
        # absent from slim images.
        if not getattr(graph_engine, "supports_cypher_queries", True):
            raise SearchTypeNotSupported(
                f"Natural language search is not supported with the "
                f"{type(graph_engine).__name__} graph backend. This retriever generates "
                "and executes Cypher queries, which require a Cypher-capable graph "
                "backend (Neo4j, Ladybug)."
            )

        is_empty = await graph_engine.is_empty()

        if is_empty:
            logger.warning("Search attempt on an empty knowledge graph")
            return []

        return await self._execute_cypher_query(query, graph_engine)

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> Optional[Any]:
        """
        Retrieves relevant context using a natural language query converted to Cypher.

        This method raises a SearchTypeNotSupported exception if the graph engine does not
        support natural language search. It also logs errors if the execution of the retrieval
        fails.

        Parameters:
        -----------

            - query (str): The natural language query used to retrieve context.

        Returns:
        --------

            - Optional[Any]: Returns the context retrieved from the graph database based on the
              query.
        """
        # Cypher rows are structured records, not prose; serialize them so the
        # context stays a plain string end-to-end (SearchResultPayload only
        # accepts str / list[str] contexts). Keep empty results falsy so
        # SearchResultPayload treats them the same as the other lexical
        # retrievers instead of surfacing a literal "[]" as the answer.
        if not retrieved_objects:
            return None
        try:
            return json.dumps(retrieved_objects, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(retrieved_objects)

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Optional[Any] = None
    ) -> Any:
        """
        Returns a completion based on the query and context.

        If context is not provided, it retrieves the context using the given query. No
        exceptions are explicitly raised from this method, but it relies on the get_context
        method for possible exceptions.

        Parameters:
        -----------

            - query (str): The natural language query to get a completion from.
            - context (Optional[Any]): The context in which to base the completion; if not
              provided, it will be retrieved using the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)

        Returns:
        --------

            - Any: Returns the completion derived from the given query and context.
        """
        # TODO: Do we want to generate a completion using LLM here?
        return context
