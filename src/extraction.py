"""
Entity and Relationship Extraction from Meditations using Groq.

Extracts structured triples (entity, relationship, entity) per passage.
"""

import json
import os
from typing import Optional
from dataclasses import dataclass

from groq import Groq


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    text: str
    type: str  # CONCEPT, PERSON, PRACTICE, STATE
    passage_id: str


@dataclass
class Relationship:
    """Represents a directed relationship between entities."""
    source_entity: str
    relationship_type: str
    target_entity: str
    passage_id: str
    confidence: float = 1.0


class EntityRelationshipExtractor:
    """
    Extracts entities and relationships from Meditations passages using Groq.
    """

    EXTRACTION_PROMPT = """
You are an expert in extracting structured knowledge from philosophical texts, specifically Marcus Aurelius's Meditations.

Given the following passage from Meditations, extract:
1. Key entities (concepts, people, practices, emotional states)
2. Relationships between these entities

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{{
  "entities": [
    {{"text": "virtue", "type": "CONCEPT"}},
    {{"text": "fear", "type": "STATE"}}
  ],
  "relationships": [
    {{"source": "fear", "type": "resolved_by", "target": "reason"}},
    {{"source": "virtue", "type": "leads_to", "target": "peace"}}
  ]
}}

Entity types: CONCEPT, PERSON, PRACTICE, STATE
Relationship types: relates_to, leads_to, teaches, resolved_by, opposes, requires, embodies

Guidelines:
- Extract 3-7 entities per passage (prioritize main concepts)
- Extract 2-5 relationships per passage
- Only include relationships explicitly or clearly implied in the passage
- Person names: use first name or common reference (Marcus, Epictetus, Socrates)
- Concepts: use lowercase, singular form (virtue not virtues)
- Be conservative: if unsure, omit

Passage:
---
{passage}
---

Return only the JSON object, no other text.
"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key. If None, uses GROQ_API_KEY env var.
        """
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = "mixtral-8x7b-32768"

    def extract(self, passage: str, passage_id: str) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract entities and relationships from a single passage.
        
        Args:
            passage: Text passage from Meditations
            passage_id: Unique identifier for the passage (e.g., "book1_ch2")
            
        Returns:
            Tuple of (entities, relationships)
        """
        prompt = self.EXTRACTION_PROMPT.format(passage=passage)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            response_text = message.content[0].text.strip()

            # Parse JSON response
            data = json.loads(response_text)

            # Convert to Entity and Relationship objects
            entities = [
                Entity(text=e["text"], type=e["type"], passage_id=passage_id)
                for e in data.get("entities", [])
            ]

            relationships = [
                Relationship(
                    source_entity=r["source"],
                    relationship_type=r["type"],
                    target_entity=r["target"],
                    passage_id=passage_id,
                )
                for r in data.get("relationships", [])
            ]

            return entities, relationships

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for passage {passage_id}: {e}")
            print(f"Response was: {response_text}")
            return [], []
        except Exception as e:
            print(f"Error extracting from passage {passage_id}: {e}")
            return [], []

    def extract_batch(self, passages: dict[str, str]) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract entities and relationships from multiple passages.
        
        Args:
            passages: Dict of {passage_id: passage_text}
            
        Returns:
            Tuple of (all_entities, all_relationships)
        """
        all_entities = []
        all_relationships = []

        for passage_id, passage_text in passages.items():
            print(f"Extracting from {passage_id}...")
            entities, relationships = self.extract(passage_text, passage_id)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        return all_entities, all_relationships


def save_extractions(
    entities: list[Entity],
    relationships: list[Relationship],
    entities_file: str,
    relationships_file: str,
):
    """
    Save extracted entities and relationships to JSON files.
    
    Args:
        entities: List of Entity objects
        relationships: List of Relationship objects
        entities_file: Path to save entities JSON
        relationships_file: Path to save relationships JSON
    """
    entities_data = [
        {
            "text": e.text,
            "type": e.type,
            "passage_id": e.passage_id,
        }
        for e in entities
    ]

    relationships_data = [
        {
            "source": r.source_entity,
            "type": r.relationship_type,
            "target": r.target_entity,
            "passage_id": r.passage_id,
            "confidence": r.confidence,
        }
        for r in relationships
    ]

    os.makedirs(os.path.dirname(entities_file), exist_ok=True)
    os.makedirs(os.path.dirname(relationships_file), exist_ok=True)

    with open(entities_file, "w") as f:
        json.dump(entities_data, f, indent=2)

    with open(relationships_file, "w") as f:
        json.dump(relationships_data, f, indent=2)

    print(f"Saved {len(entities)} entities to {entities_file}")
    print(f"Saved {len(relationships)} relationships to {relationships_file}")


def load_extractions(entities_file: str, relationships_file: str) -> tuple[list[Entity], list[Relationship]]:
    """
    Load previously extracted entities and relationships from JSON files.
    
    Args:
        entities_file: Path to entities JSON
        relationships_file: Path to relationships JSON
        
    Returns:
        Tuple of (entities, relationships)
    """
    with open(entities_file) as f:
        entities_data = json.load(f)

    with open(relationships_file) as f:
        relationships_data = json.load(f)

    entities = [
        Entity(text=e["text"], type=e["type"], passage_id=e["passage_id"])
        for e in entities_data
    ]

    relationships = [
        Relationship(
            source_entity=r["source"],
            relationship_type=r["type"],
            target_entity=r["target"],
            passage_id=r["passage_id"],
            confidence=r.get("confidence", 1.0),
        )
        for r in relationships_data
    ]

    return entities, relationships