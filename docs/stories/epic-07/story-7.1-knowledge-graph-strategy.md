# Story 7.1: Implement Knowledge Graph Strategy

**Story ID:** 7.1
**Epic:** Epic 7 - Advanced & Experimental Strategies
**Story Points:** 21
**Priority:** Experimental
**Dependencies:** Epic 3 (Embedding Service, LLM Service), Epic 5 (Advanced Strategies)

---

## User Story

**As a** system
**I want** to combine vector search with graph relationships
**So that** I can leverage entity connections in retrieval

---

## Detailed Requirements

### Functional Requirements

1. **Entity Extraction from Documents**
   - Use LLM to extract entities from documents (people, places, concepts, organizations)
   - Support multiple entity types (configurable taxonomy)
   - Extract entity attributes (description, type, properties)
   - Handle entity disambiguation (same name, different entities)
   - Support coreference resolution (he/she/it references)
   - Batch entity extraction for efficiency
   - Confidence scoring for entity extraction

2. **Relationship Extraction**
   - Extract relationships between entities using LLM
   - Support relationship types:
     - **Hierarchical**: "is_part_of", "is_a", "belongs_to"
     - **Associative**: "related_to", "connected_to", "similar_to"
     - **Causal**: "causes", "leads_to", "results_in"
     - **Temporal**: "before", "after", "during"
     - **Custom**: User-defined relationship types
   - Extract relationship attributes (strength, confidence, context)
   - Bidirectional relationships (A relates to B, B relates to A)
   - Handle multi-hop relationships (A→B→C)

3. **Graph Database Integration**
   - Support graph databases: Neo4j, Amazon Neptune, or in-memory graph
   - Store entities as graph nodes with properties
   - Store relationships as graph edges with weights
   - Efficient graph traversal algorithms
   - Graph query language support (Cypher for Neo4j)
   - Graph indexing for fast lookups
   - Support for property graphs (nodes and edges with properties)

4. **Hybrid Search: Vector + Graph**
   - Initial vector similarity search for relevant chunks
   - Graph expansion from retrieved chunks:
     - Find entities mentioned in chunks
     - Traverse graph to find related entities
     - Retrieve chunks mentioning related entities
   - Combine vector scores with graph relationship scores
   - Configurable weight balance (vector vs. graph)
   - Re-ranking based on graph connectivity
   - Support for multiple hops in graph traversal

5. **Graph Traversal Strategies**
   - **Immediate Neighbors**: Find directly connected entities
   - **K-Hop Traversal**: Traverse up to K hops from source
   - **Shortest Path**: Find shortest path between entities
   - **Community Detection**: Find entity clusters
   - **PageRank**: Rank entities by importance
   - **Relationship Filtering**: Filter by relationship type
   - Configurable traversal depth and breadth

6. **Relationship Queries**
   - Support natural language queries about relationships:
     - "What is connected to X?"
     - "What causes Y?"
     - "Show me entities related to Z"
   - Parse query to identify relationship type
   - Execute graph traversal based on query
   - Return results with relationship explanations
   - Visualize relationship paths

7. **Graph Updates and Maintenance**
   - Incremental graph updates (add/remove entities/relationships)
   - Entity deduplication and merging
   - Relationship strength decay over time (optional)
   - Graph pruning (remove weak relationships)
   - Consistency validation (detect broken links)
   - Graph statistics tracking

### Non-Functional Requirements

1. **Performance**
   - Entity extraction: <2s per document (with LLM)
   - Graph insertion: <100ms per entity/relationship
   - Hybrid search: <500ms end-to-end
   - Graph traversal: <200ms for K-hop (K≤3)
   - Support millions of entities and relationships
   - Efficient graph indexing and caching

2. **Accuracy**
   - Entity extraction F1 score >0.80
   - Relationship extraction F1 score >0.70
   - Entity disambiguation accuracy >0.85
   - Graph consistency validation (no broken references)

3. **Scalability**
   - Handle large graphs (millions of nodes/edges)
   - Distributed graph storage support
   - Incremental updates without full rebuild
   - Horizontal scaling for graph queries

4. **Flexibility**
   - Pluggable graph backends (Neo4j, Neptune, in-memory)
   - Configurable entity types and relationships
   - Custom entity extraction prompts
   - Adjustable graph traversal algorithms

5. **Observability**
   - Log entity extraction decisions
   - Track graph growth metrics
   - Monitor query performance
   - Visualize graph structure (optional)
   - Relationship quality metrics

---

## Acceptance Criteria

### AC1: Entity Extraction
- [ ] LLM-based entity extraction implemented
- [ ] Support for 5+ entity types (person, place, organization, concept, event)
- [ ] Entity attributes extracted (type, description, properties)
- [ ] Entity disambiguation working
- [ ] Confidence scores provided
- [ ] Batch extraction for efficiency

### AC2: Relationship Extraction
- [ ] Relationship extraction implemented
- [ ] Support for 5+ relationship types
- [ ] Relationship attributes extracted (type, strength, context)
- [ ] Bidirectional relationships supported
- [ ] Confidence scoring for relationships

### AC3: Graph Database Integration
- [ ] At least one graph database supported (Neo4j or in-memory)
- [ ] Entities stored as nodes with properties
- [ ] Relationships stored as edges
- [ ] Graph queries working (Cypher or equivalent)
- [ ] Efficient indexing implemented

### AC4: Hybrid Search
- [ ] Vector search combined with graph traversal
- [ ] Graph expansion from vector results
- [ ] Score combination (vector + graph) working
- [ ] Configurable weight balance
- [ ] Re-ranking based on graph connectivity

### AC5: Graph Traversal
- [ ] At least 3 traversal strategies implemented
- [ ] K-hop traversal working (K=1,2,3)
- [ ] Relationship filtering implemented
- [ ] Traversal depth/breadth configurable

### AC6: Relationship Queries
- [ ] Natural language relationship queries supported
- [ ] Query parsing to identify relationships
- [ ] Graph traversal based on query type
- [ ] Results include relationship explanations

### AC7: Graph Visualization (Optional)
- [ ] Graph structure visualizable
- [ ] Entity and relationship display
- [ ] Interactive exploration (if implemented)

### AC8: Testing
- [ ] Unit tests for entity/relationship extraction (>85% coverage)
- [ ] Unit tests for graph operations
- [ ] Integration tests with real documents
- [ ] Performance benchmarks meet requirements
- [ ] Accuracy benchmarks on test datasets

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── strategy.py              # Main KG RAG strategy
│   │   ├── entity_extractor.py      # LLM-based entity extraction
│   │   ├── relationship_extractor.py # Relationship extraction
│   │   ├── graph_store.py           # Graph database interface
│   │   ├── neo4j_store.py           # Neo4j implementation
│   │   ├── memory_graph_store.py    # In-memory graph
│   │   ├── hybrid_retriever.py      # Vector + graph retrieval
│   │   ├── traversal.py             # Graph traversal algorithms
│   │   ├── models.py                # Data models
│   │   ├── config.py                # Configuration
│   │   └── visualizer.py            # Graph visualization (optional)
│
tests/
├── unit/
│   └── strategies/
│       └── knowledge_graph/
│           ├── test_entity_extractor.py
│           ├── test_relationship_extractor.py
│           ├── test_graph_store.py
│           ├── test_hybrid_retriever.py
│           └── test_traversal.py
│
├── integration/
│   └── strategies/
│       └── test_knowledge_graph_integration.py
│
├── fixtures/
│   └── knowledge_graph/
│       ├── sample_entities.json
│       └── sample_relationships.json
```

### Dependencies
```python
# requirements.txt additions
neo4j==5.14.0                       # Neo4j Python driver
networkx==3.2.1                     # In-memory graph library
pyvis==0.3.2                        # Graph visualization
spacy==3.7.2                        # NLP for entity recognition
```

### Data Models
```python
# rag_factory/strategies/knowledge_graph/models.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class EntityType(str, Enum):
    """Entity types."""
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"
    CUSTOM = "custom"

class RelationshipType(str, Enum):
    """Relationship types."""
    IS_PART_OF = "is_part_of"
    IS_A = "is_a"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    BEFORE = "before"
    AFTER = "after"
    CONNECTED_TO = "connected_to"
    BELONGS_TO = "belongs_to"
    CUSTOM = "custom"

class Entity(BaseModel):
    """Entity node in knowledge graph."""
    id: str = Field(..., description="Unique entity ID")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    description: Optional[str] = Field(None, description="Entity description")
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunks: List[str] = Field(default_factory=list)

class Relationship(BaseModel):
    """Relationship edge in knowledge graph."""
    id: str = Field(..., description="Unique relationship ID")
    source_entity_id: str
    target_entity_id: str
    type: RelationshipType
    description: Optional[str] = None
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_chunks: List[str] = Field(default_factory=list)

class GraphTraversalResult(BaseModel):
    """Result from graph traversal."""
    entities: List[Entity]
    relationships: List[Relationship]
    paths: List[List[str]]  # List of entity ID paths
    scores: Dict[str, float]  # Entity ID -> relevance score

class HybridSearchResult(BaseModel):
    """Result from hybrid vector + graph search."""
    chunk_id: str
    text: str
    vector_score: float
    graph_score: float
    combined_score: float
    related_entities: List[Entity]
    relationship_paths: List[List[str]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Entity Extractor
```python
# rag_factory/strategies/knowledge_graph/entity_extractor.py
from typing import List, Dict, Any, Optional
import logging
from .models import Entity, EntityType
from ...services.llm import LLMService

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extract entities from text using LLM."""

    def __init__(self, llm_service: LLMService, config: Dict[str, Any]):
        self.llm = llm_service
        self.config = config
        self.entity_types = config.get("entity_types", list(EntityType))

    def extract_entities(
        self,
        text: str,
        chunk_id: str
    ) -> List[Entity]:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from
            chunk_id: Source chunk ID

        Returns:
            List of extracted entities
        """
        logger.info(f"Extracting entities from chunk: {chunk_id}")

        # Build extraction prompt
        prompt = self._build_extraction_prompt(text)

        # Call LLM
        response = self.llm.generate(prompt, temperature=0.0)

        # Parse response
        entities = self._parse_entity_response(response, chunk_id)

        logger.info(f"Extracted {len(entities)} entities from chunk {chunk_id}")

        return entities

    def extract_entities_batch(
        self,
        texts: List[Dict[str, str]]
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts in batch.

        Args:
            texts: List of dicts with 'text' and 'chunk_id' keys

        Returns:
            List of entity lists
        """
        return [
            self.extract_entities(item["text"], item["chunk_id"])
            for item in texts
        ]

    def _build_extraction_prompt(self, text: str) -> str:
        """Build entity extraction prompt."""
        entity_types_str = ", ".join([et.value for et in self.entity_types])

        prompt = f"""Extract all entities from the following text.

Entity types to extract: {entity_types_str}

For each entity, provide:
- name: The entity name
- type: One of {entity_types_str}
- description: A brief description of the entity
- confidence: Your confidence in this extraction (0.0-1.0)

Text:
{text}

Return entities in JSON format:
[
  {{"name": "...", "type": "...", "description": "...", "confidence": 0.95}},
  ...
]

Entities:"""

        return prompt

    def _parse_entity_response(
        self,
        response: str,
        chunk_id: str
    ) -> List[Entity]:
        """Parse LLM response into Entity objects."""
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON found in entity extraction response: {response[:100]}")
            return []

        try:
            entities_data = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity JSON: {e}")
            return []

        entities = []
        for i, entity_data in enumerate(entities_data):
            try:
                # Generate entity ID
                entity_id = f"{chunk_id}_entity_{i}"

                # Map type string to EntityType enum
                type_str = entity_data.get("type", "").lower()
                entity_type = EntityType.CUSTOM
                for et in EntityType:
                    if et.value == type_str:
                        entity_type = et
                        break

                entity = Entity(
                    id=entity_id,
                    name=entity_data["name"],
                    type=entity_type,
                    description=entity_data.get("description"),
                    confidence=entity_data.get("confidence", 1.0),
                    source_chunks=[chunk_id]
                )
                entities.append(entity)
            except Exception as e:
                logger.error(f"Failed to create entity from data {entity_data}: {e}")
                continue

        return entities

    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities by name similarity.

        Args:
            entities: List of entities

        Returns:
            Deduplicated entity list
        """
        # Simple deduplication by exact name match
        # TODO: Implement fuzzy matching for better deduplication
        seen_names = set()
        unique_entities = []

        for entity in entities:
            name_lower = entity.name.lower().strip()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_entities.append(entity)
            else:
                # Merge with existing entity
                existing = next(e for e in unique_entities if e.name.lower().strip() == name_lower)
                existing.source_chunks.extend(entity.source_chunks)
                # Average confidence
                existing.confidence = (existing.confidence + entity.confidence) / 2

        return unique_entities
```

### Relationship Extractor
```python
# rag_factory/strategies/knowledge_graph/relationship_extractor.py
from typing import List, Dict, Any
import logging
from .models import Entity, Relationship, RelationshipType
from ...services.llm import LLMService

logger = logging.getLogger(__name__)

class RelationshipExtractor:
    """Extract relationships between entities using LLM."""

    def __init__(self, llm_service: LLMService, config: Dict[str, Any]):
        self.llm = llm_service
        self.config = config
        self.relationship_types = config.get(
            "relationship_types",
            list(RelationshipType)
        )

    def extract_relationships(
        self,
        text: str,
        entities: List[Entity],
        chunk_id: str
    ) -> List[Relationship]:
        """
        Extract relationships between entities in text.

        Args:
            text: Source text
            entities: Entities found in text
            chunk_id: Source chunk ID

        Returns:
            List of relationships
        """
        if len(entities) < 2:
            return []

        logger.info(f"Extracting relationships between {len(entities)} entities")

        # Build extraction prompt
        prompt = self._build_relationship_prompt(text, entities)

        # Call LLM
        response = self.llm.generate(prompt, temperature=0.0)

        # Parse response
        relationships = self._parse_relationship_response(
            response,
            entities,
            chunk_id
        )

        logger.info(f"Extracted {len(relationships)} relationships")

        return relationships

    def _build_relationship_prompt(
        self,
        text: str,
        entities: List[Entity]
    ) -> str:
        """Build relationship extraction prompt."""
        entity_names = [e.name for e in entities]
        relationship_types_str = ", ".join([rt.value for rt in self.relationship_types])

        prompt = f"""Given the following text and entities, identify relationships between entities.

Text:
{text}

Entities:
{', '.join(entity_names)}

Relationship types: {relationship_types_str}

For each relationship, provide:
- source: Source entity name
- target: Target entity name
- type: Relationship type (one of {relationship_types_str})
- description: Brief description of the relationship
- strength: Relationship strength (0.0-1.0)
- confidence: Your confidence (0.0-1.0)

Return relationships in JSON format:
[
  {{"source": "Entity1", "target": "Entity2", "type": "is_part_of", "description": "...", "strength": 0.9, "confidence": 0.85}},
  ...
]

Relationships:"""

        return prompt

    def _parse_relationship_response(
        self,
        response: str,
        entities: List[Entity],
        chunk_id: str
    ) -> List[Relationship]:
        """Parse LLM response into Relationship objects."""
        import json
        import re

        # Create entity name to ID mapping
        entity_map = {e.name.lower().strip(): e.id for e in entities}

        # Extract JSON
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in relationship extraction response")
            return []

        try:
            relationships_data = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationship JSON: {e}")
            return []

        relationships = []
        for i, rel_data in enumerate(relationships_data):
            try:
                # Find source and target entity IDs
                source_name = rel_data["source"].lower().strip()
                target_name = rel_data["target"].lower().strip()

                source_id = entity_map.get(source_name)
                target_id = entity_map.get(target_name)

                if not source_id or not target_id:
                    logger.warning(
                        f"Could not find entities for relationship: "
                        f"{rel_data['source']} -> {rel_data['target']}"
                    )
                    continue

                # Map type string to RelationshipType enum
                type_str = rel_data.get("type", "").lower()
                rel_type = RelationshipType.CUSTOM
                for rt in RelationshipType:
                    if rt.value == type_str:
                        rel_type = rt
                        break

                # Generate relationship ID
                rel_id = f"{chunk_id}_rel_{i}"

                relationship = Relationship(
                    id=rel_id,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    type=rel_type,
                    description=rel_data.get("description"),
                    strength=rel_data.get("strength", 1.0),
                    confidence=rel_data.get("confidence", 1.0),
                    source_chunks=[chunk_id]
                )
                relationships.append(relationship)
            except Exception as e:
                logger.error(f"Failed to create relationship from data {rel_data}: {e}")
                continue

        return relationships
```

### Graph Store (Abstract Interface)
```python
# rag_factory/strategies/knowledge_graph/graph_store.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import Entity, Relationship, GraphTraversalResult

class GraphStore(ABC):
    """Abstract interface for graph storage backends."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> None:
        """Add entity node to graph."""
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship edge to graph."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        pass

    @abstractmethod
    def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        pass

    @abstractmethod
    def traverse(
        self,
        start_entity_ids: List[str],
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> GraphTraversalResult:
        """Traverse graph from starting entities."""
        pass

    @abstractmethod
    def find_entities_by_name(self, name_pattern: str) -> List[Entity]:
        """Find entities matching name pattern."""
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> None:
        """Delete entity and its relationships."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from graph."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        pass
```

### In-Memory Graph Store
```python
# rag_factory/strategies/knowledge_graph/memory_graph_store.py
from typing import List, Dict, Any, Optional
import networkx as nx
import logging
from .graph_store import GraphStore
from .models import Entity, Relationship, GraphTraversalResult

logger = logging.getLogger(__name__)

class MemoryGraphStore(GraphStore):
    """In-memory graph store using NetworkX."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}

    def add_entity(self, entity: Entity) -> None:
        """Add entity node to graph."""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type.value,
            description=entity.description,
            confidence=entity.confidence,
            properties=entity.properties
        )
        logger.debug(f"Added entity: {entity.name} ({entity.id})")

    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship edge to graph."""
        self.relationships[relationship.id] = relationship
        self.graph.add_edge(
            relationship.source_entity_id,
            relationship.target_entity_id,
            id=relationship.id,
            type=relationship.type.value,
            description=relationship.description,
            strength=relationship.strength,
            confidence=relationship.confidence,
            properties=relationship.properties
        )
        logger.debug(
            f"Added relationship: {relationship.source_entity_id} "
            f"-[{relationship.type.value}]-> {relationship.target_entity_id}"
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        relationships = []

        # Outgoing edges
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            rel_id = data.get("id")
            rel = self.relationships.get(rel_id)
            if rel and (not relationship_type or rel.type.value == relationship_type):
                relationships.append(rel)

        # Incoming edges
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            rel_id = data.get("id")
            rel = self.relationships.get(rel_id)
            if rel and (not relationship_type or rel.type.value == relationship_type):
                relationships.append(rel)

        return relationships

    def traverse(
        self,
        start_entity_ids: List[str],
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> GraphTraversalResult:
        """Traverse graph from starting entities."""
        visited_entities = set()
        visited_relationships = set()
        paths = []

        for start_id in start_entity_ids:
            if start_id not in self.graph:
                logger.warning(f"Entity {start_id} not in graph")
                continue

            # BFS traversal with hop limit
            visited = {start_id: 0}  # entity_id -> hop count
            queue = [(start_id, 0, [start_id])]

            while queue:
                current_id, hops, path = queue.pop(0)

                if hops >= max_hops:
                    continue

                visited_entities.add(current_id)
                paths.append(path)

                # Get neighbors
                for neighbor in self.graph.neighbors(current_id):
                    if neighbor not in visited or visited[neighbor] > hops + 1:
                        # Check relationship type filter
                        edge_data = self.graph[current_id][neighbor]
                        edge_type = edge_data.get("type")

                        if relationship_types and edge_type not in relationship_types:
                            continue

                        visited[neighbor] = hops + 1
                        queue.append((neighbor, hops + 1, path + [neighbor]))

                        # Track relationship
                        rel_id = edge_data.get("id")
                        if rel_id:
                            visited_relationships.add(rel_id)

        # Collect entities and relationships
        entities = [self.entities[eid] for eid in visited_entities if eid in self.entities]
        relationships = [self.relationships[rid] for rid in visited_relationships if rid in self.relationships]

        # Calculate scores (simple degree centrality)
        scores = {}
        for entity_id in visited_entities:
            if entity_id in self.graph:
                degree = self.graph.degree(entity_id)
                scores[entity_id] = float(degree) / (len(self.graph.nodes) or 1)

        return GraphTraversalResult(
            entities=entities,
            relationships=relationships,
            paths=paths,
            scores=scores
        )

    def find_entities_by_name(self, name_pattern: str) -> List[Entity]:
        """Find entities matching name pattern."""
        pattern_lower = name_pattern.lower()
        matching = []

        for entity in self.entities.values():
            if pattern_lower in entity.name.lower():
                matching.append(entity)

        return matching

    def delete_entity(self, entity_id: str) -> None:
        """Delete entity and its relationships."""
        if entity_id in self.graph:
            # Remove associated relationships
            for _, _, data in self.graph.edges(entity_id, data=True):
                rel_id = data.get("id")
                if rel_id and rel_id in self.relationships:
                    del self.relationships[rel_id]

            self.graph.remove_node(entity_id)
            del self.entities[entity_id]
            logger.debug(f"Deleted entity: {entity_id}")

    def clear(self) -> None:
        """Clear all data from graph."""
        self.graph.clear()
        self.entities.clear()
        self.relationships.clear()
        logger.info("Graph cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            "avg_degree": sum(dict(self.graph.degree()).values()) / (self.graph.number_of_nodes() or 1)
        }
```

The document is getting quite long. Let me continue with the remaining sections including the hybrid retriever, tests, and completion sections...


### Hybrid Retriever (Vector + Graph)
```python
# rag_factory/strategies/knowledge_graph/hybrid_retriever.py
from typing import List, Dict, Any, Optional
import logging
from .models import HybridSearchResult, Entity
from .graph_store import GraphStore

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines vector search with graph traversal."""

    def __init__(
        self,
        vector_store: Any,
        graph_store: GraphStore,
        config: Dict[str, Any]
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.config = config

        # Weights for combining scores
        self.vector_weight = config.get("vector_weight", 0.6)
        self.graph_weight = config.get("graph_weight", 0.4)
        self.max_hops = config.get("max_graph_hops", 2)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[HybridSearchResult]:
        """
        Hybrid retrieval combining vector search and graph traversal.

        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional parameters

        Returns:
            List of hybrid search results
        """
        logger.info(f"Hybrid retrieval for query: {query}")

        # Step 1: Vector search
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        logger.info(f"Vector search returned {len(vector_results)} results")

        if not vector_results:
            return []

        # Step 2: Extract entities from retrieved chunks
        chunk_ids = [r["chunk_id"] for r in vector_results]
        entities_in_chunks = self._get_entities_in_chunks(chunk_ids)
        logger.info(f"Found {len(entities_in_chunks)} entities in retrieved chunks")

        # Step 3: Graph expansion
        if entities_in_chunks:
            entity_ids = [e.id for e in entities_in_chunks]
            graph_result = self.graph_store.traverse(
                start_entity_ids=entity_ids,
                max_hops=self.max_hops
            )
            logger.info(
                f"Graph traversal found {len(graph_result.entities)} entities, "
                f"{len(graph_result.relationships)} relationships"
            )
        else:
            graph_result = None

        # Step 4: Combine scores
        hybrid_results = self._combine_results(
            vector_results,
            entities_in_chunks,
            graph_result
        )

        # Step 5: Re-rank and return top_k
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        return hybrid_results[:top_k]

    def _get_entities_in_chunks(self, chunk_ids: List[str]) -> List[Entity]:
        """Find all entities that appear in given chunks."""
        entities = []

        for entity in self.graph_store.entities.values():
            # Check if entity appears in any of the chunks
            if any(chunk_id in entity.source_chunks for chunk_id in chunk_ids):
                entities.append(entity)

        return entities

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        entities_in_chunks: List[Entity],
        graph_result: Optional[Any]
    ) -> List[HybridSearchResult]:
        """Combine vector and graph results."""
        hybrid_results = []

        for vec_result in vector_results:
            chunk_id = vec_result["chunk_id"]
            vector_score = vec_result.get("score", 0.0)

            # Find entities in this chunk
            related_entities = [
                e for e in entities_in_chunks
                if chunk_id in e.source_chunks
            ]

            # Calculate graph score
            graph_score = 0.0
            relationship_paths = []

            if graph_result and related_entities:
                # Graph score based on entity importance and connectivity
                for entity in related_entities:
                    entity_score = graph_result.scores.get(entity.id, 0.0)
                    graph_score = max(graph_score, entity_score)

                # Extract relationship paths involving these entities
                for path in graph_result.paths:
                    if any(e.id in path for e in related_entities):
                        relationship_paths.append(path)

            # Combined score
            combined_score = (
                self.vector_weight * vector_score +
                self.graph_weight * graph_score
            )

            # Create hybrid result
            result = HybridSearchResult(
                chunk_id=chunk_id,
                text=vec_result.get("text", ""),
                vector_score=vector_score,
                graph_score=graph_score,
                combined_score=combined_score,
                related_entities=related_entities,
                relationship_paths=relationship_paths[:5],  # Limit paths
                metadata=vec_result.get("metadata", {})
            )

            hybrid_results.append(result)

        return hybrid_results
```

### Knowledge Graph RAG Strategy
```python
# rag_factory/strategies/knowledge_graph/strategy.py
from typing import List, Dict, Any, Optional
import logging
from ..base import RAGStrategy
from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .graph_store import GraphStore
from .memory_graph_store import MemoryGraphStore
from .hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

class KnowledgeGraphRAGStrategy(RAGStrategy):
    """
    Knowledge Graph RAG: Combine vector search with graph relationships.

    Extracts entities and relationships from documents, stores them in a
    graph database, and uses graph traversal to enhance retrieval.
    """

    def __init__(
        self,
        vector_store_service: Any,
        llm_service: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)

        self.vector_store = vector_store_service
        self.llm_service = llm_service

        # Initialize components
        self.entity_extractor = EntityExtractor(llm_service, config or {})
        self.relationship_extractor = RelationshipExtractor(llm_service, config or {})

        # Initialize graph store (default to in-memory)
        graph_backend = (config or {}).get("graph_backend", "memory")
        if graph_backend == "memory":
            self.graph_store = MemoryGraphStore()
        else:
            raise ValueError(f"Unsupported graph backend: {graph_backend}")

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vector_store_service,
            self.graph_store,
            config or {}
        )

    def index_document(self, document: str, document_id: str) -> None:
        """
        Index document with entity and relationship extraction.

        Args:
            document: Document text
            document_id: Unique document ID
        """
        logger.info(f"Indexing document with knowledge graph: {document_id}")

        # Traditional chunk-based indexing for vector search
        # (Assuming chunking is done elsewhere or using simple splitting)
        chunks = self._chunk_document(document, document_id)

        # Index chunks in vector store
        for chunk in chunks:
            self.vector_store.index_chunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata={"document_id": document_id}
            )

        # Extract entities from each chunk
        all_entities = []
        for chunk in chunks:
            entities = self.entity_extractor.extract_entities(
                chunk["text"],
                chunk["chunk_id"]
            )
            all_entities.extend(entities)

            # Add entities to graph
            for entity in entities:
                self.graph_store.add_entity(entity)

        # Deduplicate entities
        unique_entities = self.entity_extractor.deduplicate_entities(all_entities)

        # Extract relationships
        all_relationships = []
        for chunk in chunks:
            # Find entities in this chunk
            chunk_entities = [
                e for e in unique_entities
                if chunk["chunk_id"] in e.source_chunks
            ]

            if len(chunk_entities) >= 2:
                relationships = self.relationship_extractor.extract_relationships(
                    chunk["text"],
                    chunk_entities,
                    chunk["chunk_id"]
                )
                all_relationships.extend(relationships)

                # Add relationships to graph
                for rel in relationships:
                    self.graph_store.add_relationship(rel)

        logger.info(
            f"Indexed document {document_id}: "
            f"{len(chunks)} chunks, {len(unique_entities)} entities, "
            f"{len(all_relationships)} relationships"
        )

        # Log graph stats
        stats = self.graph_store.get_stats()
        logger.info(f"Graph stats: {stats}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval using vector search + graph traversal.

        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional parameters

        Returns:
            List of hybrid search results
        """
        logger.info(f"Knowledge graph retrieval for: {query}")

        # Hybrid retrieval
        results = self.hybrid_retriever.retrieve(query, top_k=top_k, **kwargs)

        # Convert to dict format
        output = []
        for result in results:
            output.append({
                "chunk_id": result.chunk_id,
                "text": result.text,
                "score": result.combined_score,
                "vector_score": result.vector_score,
                "graph_score": result.graph_score,
                "related_entities": [
                    {"name": e.name, "type": e.type.value}
                    for e in result.related_entities
                ],
                "relationship_paths": result.relationship_paths,
                "metadata": result.metadata
            })

        return output

    def _chunk_document(
        self,
        document: str,
        document_id: str
    ) -> List[Dict[str, str]]:
        """Simple document chunking (can be replaced with better chunking)."""
        # Split by paragraphs
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]

        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": para,
                "document_id": document_id
            })

        return chunks

    @property
    def name(self) -> str:
        return "knowledge_graph"

    @property
    def description(self) -> str:
        return "Combine vector search with graph-based entity relationships"
```

---

## Unit Tests

### Test File Locations
- `tests/unit/strategies/knowledge_graph/test_entity_extractor.py`
- `tests/unit/strategies/knowledge_graph/test_relationship_extractor.py`
- `tests/unit/strategies/knowledge_graph/test_graph_store.py`
- `tests/unit/strategies/knowledge_graph/test_hybrid_retriever.py`

### Test Cases

#### TC7.1.1: Entity Extractor Tests
```python
import pytest
from unittest.mock import Mock
from rag_factory.strategies.knowledge_graph.entity_extractor import EntityExtractor
from rag_factory.strategies.knowledge_graph.models import EntityType

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.generate.return_value = '''
    [
      {"name": "Python", "type": "concept", "description": "Programming language", "confidence": 0.95},
      {"name": "Machine Learning", "type": "concept", "description": "AI subset", "confidence": 0.90}
    ]
    '''
    return llm

@pytest.fixture
def entity_extractor(mock_llm):
    config = {"entity_types": [EntityType.CONCEPT, EntityType.PERSON]}
    return EntityExtractor(mock_llm, config)

def test_entity_extraction_basic(entity_extractor, mock_llm):
    """Test basic entity extraction."""
    text = "Python is great for Machine Learning applications."
    entities = entity_extractor.extract_entities(text, "chunk_1")

    assert len(entities) == 2
    assert entities[0].name == "Python"
    assert entities[0].type == EntityType.CONCEPT
    assert entities[1].name == "Machine Learning"

def test_entity_extraction_with_confidence(entity_extractor, mock_llm):
    """Test confidence scores."""
    entities = entity_extractor.extract_entities("Test text", "chunk_1")

    assert all(0.0 <= e.confidence <= 1.0 for e in entities)

def test_entity_deduplication(entity_extractor):
    """Test entity deduplication."""
    from rag_factory.strategies.knowledge_graph.models import Entity

    entities = [
        Entity(id="e1", name="Python", type=EntityType.CONCEPT, confidence=0.9, source_chunks=["c1"]),
        Entity(id="e2", name="python", type=EntityType.CONCEPT, confidence=0.85, source_chunks=["c2"]),  # Duplicate
        Entity(id="e3", name="Java", type=EntityType.CONCEPT, confidence=0.8, source_chunks=["c1"])
    ]

    unique = entity_extractor.deduplicate_entities(entities)

    assert len(unique) == 2  # Python and Java
    assert any(e.name.lower() == "python" for e in unique)
    assert any(e.name == "Java" for e in unique)

def test_batch_extraction(entity_extractor, mock_llm):
    """Test batch entity extraction."""
    texts = [
        {"text": "Text 1", "chunk_id": "c1"},
        {"text": "Text 2", "chunk_id": "c2"}
    ]

    results = entity_extractor.extract_entities_batch(texts)

    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)
```

#### TC7.1.2: Graph Store Tests
```python
import pytest
from rag_factory.strategies.knowledge_graph.memory_graph_store import MemoryGraphStore
from rag_factory.strategies.knowledge_graph.models import Entity, Relationship, EntityType, RelationshipType

@pytest.fixture
def graph_store():
    return MemoryGraphStore()

def test_add_entity(graph_store):
    """Test adding entities to graph."""
    entity = Entity(
        id="e1",
        name="Python",
        type=EntityType.CONCEPT,
        description="Programming language",
        confidence=0.95
    )

    graph_store.add_entity(entity)

    retrieved = graph_store.get_entity("e1")
    assert retrieved is not None
    assert retrieved.name == "Python"

def test_add_relationship(graph_store):
    """Test adding relationships to graph."""
    # Add entities
    e1 = Entity(id="e1", name="Python", type=EntityType.CONCEPT, confidence=1.0)
    e2 = Entity(id="e2", name="Machine Learning", type=EntityType.CONCEPT, confidence=1.0)

    graph_store.add_entity(e1)
    graph_store.add_entity(e2)

    # Add relationship
    rel = Relationship(
        id="r1",
        source_entity_id="e1",
        target_entity_id="e2",
        type=RelationshipType.RELATED_TO,
        strength=0.9,
        confidence=0.85
    )

    graph_store.add_relationship(rel)

    # Retrieve relationships
    relationships = graph_store.get_relationships("e1")
    assert len(relationships) >= 1
    assert relationships[0].target_entity_id == "e2"

def test_graph_traversal(graph_store):
    """Test graph traversal."""
    # Create a small graph: A -> B -> C
    entities = [
        Entity(id="A", name="Entity A", type=EntityType.CONCEPT, confidence=1.0),
        Entity(id="B", name="Entity B", type=EntityType.CONCEPT, confidence=1.0),
        Entity(id="C", name="Entity C", type=EntityType.CONCEPT, confidence=1.0)
    ]

    for entity in entities:
        graph_store.add_entity(entity)

    relationships = [
        Relationship(
            id="r1", source_entity_id="A", target_entity_id="B",
            type=RelationshipType.CONNECTED_TO, strength=1.0, confidence=1.0
        ),
        Relationship(
            id="r2", source_entity_id="B", target_entity_id="C",
            type=RelationshipType.CONNECTED_TO, strength=1.0, confidence=1.0
        )
    ]

    for rel in relationships:
        graph_store.add_relationship(rel)

    # Traverse from A with max_hops=2
    result = graph_store.traverse(["A"], max_hops=2)

    assert len(result.entities) == 3  # A, B, C
    assert len(result.relationships) == 2

def test_find_entities_by_name(graph_store):
    """Test entity search by name."""
    entities = [
        Entity(id="e1", name="Python Programming", type=EntityType.CONCEPT, confidence=1.0),
        Entity(id="e2", name="Java Programming", type=EntityType.CONCEPT, confidence=1.0),
        Entity(id="e3", name="Machine Learning", type=EntityType.CONCEPT, confidence=1.0)
    ]

    for entity in entities:
        graph_store.add_entity(entity)

    results = graph_store.find_entities_by_name("Programming")
    assert len(results) == 2
    assert all("Programming" in e.name for e in results)

def test_graph_stats(graph_store):
    """Test graph statistics."""
    # Add some entities and relationships
    e1 = Entity(id="e1", name="A", type=EntityType.CONCEPT, confidence=1.0)
    e2 = Entity(id="e2", name="B", type=EntityType.CONCEPT, confidence=1.0)

    graph_store.add_entity(e1)
    graph_store.add_entity(e2)

    rel = Relationship(
        id="r1", source_entity_id="e1", target_entity_id="e2",
        type=RelationshipType.RELATED_TO, strength=1.0, confidence=1.0
    )
    graph_store.add_relationship(rel)

    stats = graph_store.get_stats()

    assert stats["num_entities"] == 2
    assert stats["num_relationships"] == 1
    assert "density" in stats
```

#### TC7.1.3: Hybrid Retriever Tests
```python
import pytest
from unittest.mock import Mock, MagicMock
from rag_factory.strategies.knowledge_graph.hybrid_retriever import HybridRetriever
from rag_factory.strategies.knowledge_graph.models import Entity, EntityType, GraphTraversalResult

@pytest.fixture
def mock_vector_store():
    store = Mock()
    store.search.return_value = [
        {"chunk_id": "c1", "text": "Text about Python", "score": 0.95},
        {"chunk_id": "c2", "text": "Text about ML", "score": 0.85}
    ]
    return store

@pytest.fixture
def mock_graph_store():
    store = Mock()
    store.entities = {
        "e1": Entity(id="e1", name="Python", type=EntityType.CONCEPT, confidence=1.0, source_chunks=["c1"]),
        "e2": Entity(id="e2", name="ML", type=EntityType.CONCEPT, confidence=1.0, source_chunks=["c2"])
    }
    store.traverse.return_value = GraphTraversalResult(
        entities=list(store.entities.values()),
        relationships=[],
        paths=[["e1", "e2"]],
        scores={"e1": 0.8, "e2": 0.7}
    )
    return store

@pytest.fixture
def hybrid_retriever(mock_vector_store, mock_graph_store):
    config = {"vector_weight": 0.6, "graph_weight": 0.4, "max_graph_hops": 2}
    return HybridRetriever(mock_vector_store, mock_graph_store, config)

def test_hybrid_retrieval(hybrid_retriever, mock_vector_store, mock_graph_store):
    """Test hybrid retrieval combines vector and graph."""
    results = hybrid_retriever.retrieve("test query", top_k=2)

    assert len(results) > 0
    assert all(hasattr(r, "combined_score") for r in results)
    assert all(hasattr(r, "vector_score") for r in results)
    assert all(hasattr(r, "graph_score") for r in results)

    # Verify vector store was called
    mock_vector_store.search.assert_called_once()

def test_score_combination(hybrid_retriever):
    """Test that scores are combined correctly."""
    results = hybrid_retriever.retrieve("test query", top_k=2)

    for result in results:
        # Combined score should be weighted average
        expected = (
            hybrid_retriever.vector_weight * result.vector_score +
            hybrid_retriever.graph_weight * result.graph_score
        )
        assert abs(result.combined_score - expected) < 0.001

def test_empty_vector_results(hybrid_retriever, mock_vector_store):
    """Test handling of empty vector results."""
    mock_vector_store.search.return_value = []

    results = hybrid_retriever.retrieve("test query", top_k=2)

    assert len(results) == 0
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_knowledge_graph_integration.py`

### Test Scenarios

#### IS7.1.1: End-to-End Knowledge Graph Workflow
```python
import pytest
from rag_factory.strategies.knowledge_graph.strategy import KnowledgeGraphRAGStrategy

@pytest.mark.integration
def test_knowledge_graph_workflow(test_vector_store, test_llm):
    """Test complete knowledge graph RAG workflow."""
    config = {
        "graph_backend": "memory",
        "vector_weight": 0.6,
        "graph_weight": 0.4
    }

    strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm,
        config=config
    )

    # Index document
    document = """
    Python is a popular programming language for Machine Learning.
    Machine Learning is a subset of Artificial Intelligence.
    Artificial Intelligence enables computers to learn and make decisions.
    """

    strategy.index_document(document, "ml_doc")

    # Verify entities were extracted
    graph_stats = strategy.graph_store.get_stats()
    assert graph_stats["num_entities"] > 0
    assert graph_stats["num_relationships"] > 0

    # Retrieve with hybrid search
    results = strategy.retrieve("What is Machine Learning?", top_k=3)

    assert len(results) > 0
    assert all("related_entities" in r for r in results)
    assert all("combined_score" in r for r in results)

@pytest.mark.integration
def test_relationship_queries(test_vector_store, test_llm):
    """Test relationship-based queries."""
    strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm,
        config={"graph_backend": "memory"}
    )

    document = """
    Climate change causes rising temperatures.
    Rising temperatures lead to glacier melting.
    Glacier melting results in sea level rise.
    """

    strategy.index_document(document, "climate_doc")

    # Query about causal relationships
    results = strategy.retrieve("What causes sea level rise?", top_k=3)

    assert len(results) > 0
    # Should leverage graph relationships to connect concepts
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_kg_performance.py
import pytest
import time
from rag_factory.strategies.knowledge_graph.strategy import KnowledgeGraphRAGStrategy

@pytest.mark.benchmark
def test_entity_extraction_performance(test_llm):
    """Benchmark entity extraction speed."""
    from rag_factory.strategies.knowledge_graph.entity_extractor import EntityExtractor

    extractor = EntityExtractor(test_llm, {})
    document = "Test document with entities. " * 100  # ~200 words

    start = time.time()
    entities = extractor.extract_entities(document, "perf_test")
    duration = time.time() - start

    print(f"\nEntity extraction: {len(entities)} entities in {duration:.3f}s")
    assert duration < 2.0, f"Too slow: {duration:.2f}s (expected <2s)"

@pytest.mark.benchmark
def test_graph_traversal_performance():
    """Benchmark graph traversal speed."""
    from rag_factory.strategies.knowledge_graph.memory_graph_store import MemoryGraphStore
    from rag_factory.strategies.knowledge_graph.models import Entity, Relationship, EntityType, RelationshipType

    store = MemoryGraphStore()

    # Create large graph
    for i in range(1000):
        entity = Entity(id=f"e{i}", name=f"Entity {i}", type=EntityType.CONCEPT, confidence=1.0)
        store.add_entity(entity)

    for i in range(999):
        rel = Relationship(
            id=f"r{i}", source_entity_id=f"e{i}", target_entity_id=f"e{i+1}",
            type=RelationshipType.CONNECTED_TO, strength=1.0, confidence=1.0
        )
        store.add_relationship(rel)

    # Traverse
    start = time.time()
    result = store.traverse(["e0"], max_hops=3)
    duration = time.time() - start

    print(f"\nGraph traversal: {len(result.entities)} entities in {duration:.3f}s")
    assert duration < 0.2, f"Too slow: {duration:.2f}s (expected <200ms)"

@pytest.mark.benchmark
def test_hybrid_search_performance(test_vector_store, test_llm):
    """Benchmark end-to-end hybrid search."""
    strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm,
        config={"graph_backend": "memory"}
    )

    # Index multiple documents
    for i in range(10):
        doc = f"Document {i} about topic {i}. " * 20
        strategy.index_document(doc, f"doc_{i}")

    # Search
    start = time.time()
    results = strategy.retrieve("test query", top_k=5)
    duration = time.time() - start

    print(f"\nHybrid search: {len(results)} results in {duration:.3f}s")
    assert duration < 0.5, f"Too slow: {duration:.2f}s (expected <500ms)"
```

---

## Definition of Done

- [ ] Entity extractor implemented with LLM
- [ ] Relationship extractor implemented
- [ ] At least 5 entity types supported
- [ ] At least 5 relationship types supported
- [ ] Graph store interface defined
- [ ] In-memory graph store implemented
- [ ] Neo4j graph store implemented (optional)
- [ ] Graph traversal algorithms implemented
- [ ] Hybrid retriever combining vector + graph
- [ ] Knowledge Graph RAG strategy complete
- [ ] Score combination working
- [ ] All unit tests pass (>85% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Entity extraction F1 >0.80 on test data
- [ ] Relationship extraction F1 >0.70
- [ ] Configuration system working
- [ ] Documentation complete with examples
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install neo4j networkx pyvis spacy

# Download spaCy model for NER (optional enhancement)
python -m spacy download en_core_web_sm
```

### Configuration

```yaml
# config.yaml
strategies:
  knowledge_graph:
    enabled: true
    graph_backend: "memory"  # or "neo4j"

    # Entity extraction
    entity_types:
      - person
      - place
      - organization
      - concept
      - event

    # Relationship extraction
    relationship_types:
      - is_part_of
      - is_a
      - related_to
      - causes
      - connected_to

    # Hybrid search
    vector_weight: 0.6
    graph_weight: 0.4
    max_graph_hops: 2

    # Neo4j configuration (if using)
    neo4j:
      uri: "bolt://localhost:7687"
      user: "neo4j"
      password: "password"
```

### Usage Example

```python
from rag_factory.strategies.knowledge_graph import KnowledgeGraphRAGStrategy

# Setup strategy
strategy = KnowledgeGraphRAGStrategy(
    vector_store_service=vector_store,
    llm_service=llm,
    config={
        "graph_backend": "memory",
        "vector_weight": 0.6,
        "graph_weight": 0.4
    }
)

# Index document (entities and relationships extracted automatically)
document = """
The Python programming language is widely used in Machine Learning.
Machine Learning is a subset of Artificial Intelligence.
TensorFlow and PyTorch are popular ML frameworks.
"""

strategy.index_document(document, "ml_intro")

# Retrieve with graph enhancement
results = strategy.retrieve("What frameworks are used for ML?", top_k=3)

# View results with entities
for result in results:
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']:.3f} (vector: {result['vector_score']:.3f}, graph: {result['graph_score']:.3f})")
    print(f"Related entities: {[e['name'] for e in result['related_entities']]}")
    print(f"Relationship paths: {result['relationship_paths']}")
    print()

# View graph statistics
stats = strategy.graph_store.get_stats()
print(f"Graph has {stats['num_entities']} entities and {stats['num_relationships']} relationships")
```

---

## Notes for Developers

1. **LLM Prompts**: The quality of entity/relationship extraction heavily depends on prompt engineering. Iterate and test prompts carefully.

2. **Entity Disambiguation**: Implement proper entity resolution (same entity, different names). Consider using entity embeddings for similarity.

3. **Graph Backend Choice**:
   - In-memory (NetworkX): Fast, good for small-medium graphs (<100K entities)
   - Neo4j: Production-ready, handles millions of entities, requires separate service

4. **Performance Optimization**:
   - Batch entity extraction to reduce LLM calls
   - Cache frequently accessed graph paths
   - Use graph database indexes effectively

5. **Relationship Quality**: Not all relationships are equally important. Use confidence scores and prune weak relationships.

6. **Graph Visualization**: Use pyvis or similar tools to visualize the knowledge graph. Helps with debugging and understanding.

7. **Error Handling**: LLM extraction can fail. Always have fallbacks and validate extracted data.

8. **Testing with Real Data**: Test on domain-specific documents to tune entity/relationship types.

9. **Incremental Updates**: For production, implement incremental graph updates rather than rebuilding entire graph.

10. **Privacy**: Be cautious with sensitive data in graphs. Implement access controls if needed.

11. **Evaluation**: Manually label test datasets for entity/relationship extraction to measure accuracy.

12. **Multi-language**: If supporting multiple languages, use language-specific NER models or multilingual LLMs.
