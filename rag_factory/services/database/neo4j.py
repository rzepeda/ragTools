"""Neo4j graph database service implementation.

This module provides a graph database service that implements IGraphService
using Neo4j.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import logging

try:
    from neo4j import AsyncGraphDatabase
    if TYPE_CHECKING:
        from neo4j import AsyncDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncDriver = Any  # type: ignore

from rag_factory.services.interfaces import IGraphService

logger = logging.getLogger(__name__)


class Neo4jGraphService(IGraphService):
    """Neo4j graph database service.

    This service implements IGraphService using Neo4j for graph database
    operations used in knowledge graph RAG strategies.

    Example:
        >>> service = Neo4jGraphService(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password"
        ... )
        >>> node_id = await service.create_node("Person", {"name": "Alice"})
        >>> await service.create_relationship(
        ...     node_id, other_node_id, "KNOWS", {"since": 2020}
        ... )
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j"
    ):
        """Initialize Neo4j graph service.

        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            user: Database username
            password: Database password
            database: Database name (default: neo4j)

        Raises:
            ImportError: If neo4j package is not installed
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j package not installed. Install with:\n"
                "  pip install neo4j"
            )

        self.uri = uri
        self.database = database
        self._driver: Optional["AsyncDriver"] = None  # type: ignore
        self._auth = (user, password)

        logger.info(f"Initialized Neo4j service for {uri}")

    async def _get_driver(self) -> "AsyncDriver":  # type: ignore
        """Get or create async driver.

        Returns:
            Neo4j async driver instance
        """
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=self._auth
            )
        return self._driver

    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> str:
        """Create a node in the graph.

        Args:
            label: Node label/type (e.g., "Entity", "Document", "Person")
            properties: Dictionary of node properties

        Returns:
            Unique identifier for the created node

        Raises:
            Exception: If node creation fails
        """
        driver = await self._get_driver()

        async with driver.session(database=self.database) as session:
            result = await session.run(
                f"CREATE (n:{label} $props) RETURN elementId(n) as id",
                props=properties
            )
            record = await result.single()
            node_id = record["id"]

            logger.debug(f"Created node {label} with id {node_id}")
            return node_id

    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create relationship between nodes.

        Args:
            from_node_id: ID of the source node
            to_node_id: ID of the target node
            relationship_type: Type of relationship (e.g., "RELATES_TO", "MENTIONS")
            properties: Optional dictionary of relationship properties

        Raises:
            Exception: If relationship creation fails or nodes don't exist
        """
        driver = await self._get_driver()

        props = properties or {}

        async with driver.session(database=self.database) as session:
            await session.run(
                f"""
                MATCH (a), (b)
                WHERE elementId(a) = $from_id AND elementId(b) = $to_id
                CREATE (a)-[r:{relationship_type} $props]->(b)
                """,
                from_id=from_node_id,
                to_id=to_node_id,
                props=props
            )

            logger.debug(
                f"Created relationship {relationship_type} "
                f"from {from_node_id} to {to_node_id}"
            )

    async def query(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query.

        Args:
            cypher_query: Cypher query string to execute
            parameters: Optional parameters for the query (for parameterized queries)

        Returns:
            List of result records as dictionaries

        Raises:
            Exception: If query execution fails or query is invalid
        """
        driver = await self._get_driver()

        params = parameters or {}

        async with driver.session(database=self.database) as session:
            result = await session.run(cypher_query, params)
            records = await result.data()

            logger.debug(f"Executed query, returned {len(records)} records")
            return records

    async def close(self):
        """Close the database connection.

        Should be called when the service is no longer needed.
        """
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Closed Neo4j connection")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
