"""
Relationship extraction from text using LLM.

This module implements LLM-based relationship extraction between entities
for the knowledge graph strategy.
"""

from typing import List, Dict, Any
import logging
import json
import re

from .models import Entity, Relationship, RelationshipType
from .config import KnowledgeGraphConfig

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """
    Extract relationships between entities using LLM.
    
    Uses structured prompts to identify relationships between entities
    in text with types, descriptions, strength, and confidence scores.
    """

    def __init__(self, llm_service: Any, config: KnowledgeGraphConfig):
        """
        Initialize relationship extractor.
        
        Args:
            llm_service: LLM service for relationship extraction
            config: Knowledge graph configuration
        """
        self.llm = llm_service
        self.config = config
        self.relationship_types = config.relationship_types

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
        from ...services.llm.base import Message, MessageRole
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = self.llm.complete(messages, temperature=0.0, max_tokens=2000)
        
        # Parse response
        relationships = self._parse_relationship_response(
            response.content,
            entities,
            chunk_id
        )
        
        # Filter by confidence and strength
        relationships = [
            r for r in relationships
            if r.confidence >= self.config.min_relationship_confidence
            and r.strength >= self.config.min_relationship_strength
        ]
        
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
        # Create entity name to ID mapping (case-insensitive)
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
