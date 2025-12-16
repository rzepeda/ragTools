"""
Entity extraction from text using LLM.

This module implements LLM-based entity extraction for the knowledge graph strategy.
"""

from typing import List, Dict, Any
import logging
import json
import re

from .models import Entity, EntityType
from .config import KnowledgeGraphConfig

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extract entities from text using LLM.
    
    Uses structured prompts to extract entities with types, descriptions,
    and confidence scores from text chunks.
    """

    def __init__(self, llm_service: Any, config: KnowledgeGraphConfig):
        """
        Initialize entity extractor.
        
        Args:
            llm_service: LLM service for entity extraction
            config: Knowledge graph configuration
        """
        self.llm = llm_service
        self.config = config
        self.entity_types = config.entity_types

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
        from ...services.llm.base import Message, MessageRole
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = self.llm.complete(messages, temperature=0.0, max_tokens=2000)
        
        # Parse response
        entities = self._parse_entity_response(response.content, chunk_id)
        
        # Filter by confidence
        entities = [
            e for e in entities
            if e.confidence >= self.config.min_entity_confidence
        ]
        
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
        if not self.config.enable_entity_deduplication:
            return entities
        
        # Simple deduplication by exact name match (case-insensitive)
        seen_names = {}
        unique_entities = []
        
        for entity in entities:
            name_key = (entity.name.lower().strip(), entity.type.value)
            
            if name_key not in seen_names:
                seen_names[name_key] = len(unique_entities)
                unique_entities.append(entity)
            else:
                # Merge with existing entity
                idx = seen_names[name_key]
                existing = unique_entities[idx]
                
                # Extend source chunks
                existing.source_chunks.extend(entity.source_chunks)
                existing.source_chunks = list(set(existing.source_chunks))
                
                # Average confidence
                existing.confidence = (existing.confidence + entity.confidence) / 2
        
        logger.info(f"Deduplicated {len(entities)} entities to {len(unique_entities)}")
        
        return unique_entities
