"""
Entity Deduplication and Linking.

Consolidates entity mentions across passages using embedding similarity and fuzzy matching.
Maps "virtue" mentions → canonical entity "virtue".
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer

from extraction import Entity, Relationship


@dataclass
class CanonicalEntity:
    """Represents a deduplicated entity with variants."""
    canonical_text: str
    entity_type: str
    variants: list[str] = field(default_factory=list)
    passage_ids: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class EntityDeduplicator:
    """
    Deduplicates entities using embedding similarity and fuzzy matching.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        fuzzy_threshold: float = 0.80,
    ):
        """
        Initialize deduplicator.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            similarity_threshold: Embedding similarity threshold (0-1)
            fuzzy_threshold: Fuzzy match threshold (0-1)
        """
        self.model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold

    def _fuzzy_match(self, text1: str, text2: str) -> float:
        """Compute fuzzy string similarity."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def deduplicate(self, entities: list[Entity]) -> dict[str, CanonicalEntity]:
        """
        Deduplicate entities into canonical forms.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dict mapping canonical_text -> CanonicalEntity
        """
        if not entities:
            return {}

        # Group by entity type (don't link "Marcus" PERSON with "Marcus" CONCEPT)
        entities_by_type = {}
        for entity in entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)

        canonical_map = {}  # {canonical_text -> CanonicalEntity}

        # Process each entity type separately
        for entity_type, type_entities in entities_by_type.items():
            print(f"\nDeduplicating {entity_type} entities ({len(type_entities)} total)...")

            # Get unique texts
            unique_texts = list(set(e.text for e in type_entities))

            # Compute embeddings
            embeddings = self.model.encode(unique_texts, convert_to_numpy=True)
            embedding_map = {text: emb for text, emb in zip(unique_texts, embeddings)}

            # Clustering: merge similar entities
            canonical_to_variants = {}
            processed = set()

            for text in unique_texts:
                if text in processed:
                    continue

                canonical = text
                variants = [text]
                processed.add(text)

                emb1 = embedding_map[text]

                # Find similar entities
                for other_text in unique_texts:
                    if other_text == text or other_text in processed:
                        continue

                    emb2 = embedding_map[other_text]

                    # Check both embedding and fuzzy similarity
                    emb_sim = self._embedding_similarity(emb1, emb2)
                    fuzzy_sim = self._fuzzy_match(text, other_text)

                    if emb_sim >= self.similarity_threshold or fuzzy_sim >= self.fuzzy_threshold:
                        variants.append(other_text)
                        processed.add(other_text)

                canonical_to_variants[canonical] = variants

            # Create CanonicalEntity objects
            for canonical, variants in canonical_to_variants.items():
                passage_ids = list(
                    set(
                        e.passage_id
                        for e in type_entities
                        if e.text in variants
                    )
                )

                canonical_map[canonical] = CanonicalEntity(
                    canonical_text=canonical,
                    entity_type=entity_type,
                    variants=variants,
                    passage_ids=passage_ids,
                    embedding=embedding_map[canonical],
                )

            print(f"Reduced {len(unique_texts)} to {len(canonical_to_variants)} canonical entities")

        return canonical_map

    def remap_relationships(
        self,
        relationships: list[Relationship],
        canonical_map: dict[str, CanonicalEntity],
    ) -> list[Relationship]:
        """
        Remap relationship entities to canonical forms.
        
        Args:
            relationships: List of extracted relationships
            canonical_map: Mapping from canonical entity text to CanonicalEntity
            
        Returns:
            List of remapped relationships
        """
        # Build reverse map: variant -> canonical
        variant_to_canonical = {}
        for canonical_entity in canonical_map.values():
            for variant in canonical_entity.variants:
                variant_to_canonical[variant] = canonical_entity.canonical_text

        remapped = []
        for rel in relationships:
            source_canon = variant_to_canonical.get(rel.source_entity, rel.source_entity)
            target_canon = variant_to_canonical.get(rel.target_entity, rel.target_entity)

            # Only keep relationships where both entities are in canonical map
            if source_canon in canonical_map and target_canon in canonical_map:
                rel.source_entity = source_canon
                rel.target_entity = target_canon
                remapped.append(rel)

        print(f"Remapped {len(relationships)} relationships, kept {len(remapped)}")
        return remapped


def save_canonical_entities(
    canonical_map: dict[str, CanonicalEntity],
    output_file: str,
):
    """
    Save canonical entities to JSON.
    
    Args:
        canonical_map: Mapping from canonical text to CanonicalEntity
        output_file: Path to save JSON
    """
    data = {
        canonical_text: {
            "canonical_text": entity.canonical_text,
            "type": entity.entity_type,
            "variants": entity.variants,
            "passage_ids": entity.passage_ids,
            "count": len(entity.passage_ids),
        }
        for canonical_text, entity in canonical_map.items()
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(canonical_map)} canonical entities to {output_file}")


def load_canonical_entities(input_file: str) -> dict[str, CanonicalEntity]:
    """
    Load canonical entities from JSON.
    
    Args:
        input_file: Path to canonical entities JSON
        
    Returns:
        Mapping from canonical text to CanonicalEntity
    """
    with open(input_file) as f:
        data = json.load(f)

    canonical_map = {
        canonical_text: CanonicalEntity(
            canonical_text=entity["canonical_text"],
            entity_type=entity["type"],
            variants=entity["variants"],
            passage_ids=entity["passage_ids"],
        )
        for canonical_text, entity in data.items()
    }

    return canonical_map