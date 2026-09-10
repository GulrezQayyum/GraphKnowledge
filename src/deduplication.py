import json
from typing import Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .extraction import Entity, Relationship
except ImportError:
    from extraction import Entity, Relationship


@dataclass
class CanonicalEntity:
    canonical_text: str
    entity_type: str
    variants: list[str] = field(default_factory=list)
    passage_ids: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class EntityDeduplicator:

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        fuzzy_threshold: float = 0.80,
        model: Optional[SentenceTransformer] = None,
    ):
        self.model = model or SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold

    def _fuzzy_match(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def deduplicate(self, entities: list[Entity]) -> dict[str, CanonicalEntity]:
       
        if not entities:
            return {}

        entities_by_type = {}
        for entity in entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)

        canonical_map = {}  
        for entity_type, type_entities in entities_by_type.items():
            print(f"\nDeduplicating {entity_type} entities ({len(type_entities)} total)...")

            unique_texts = sorted({e.text for e in type_entities})

            embeddings = self.model.encode(unique_texts, convert_to_numpy=True)
            embedding_map = {text: emb for text, emb in zip(unique_texts, embeddings)}

            canonical_to_variants = {}
            processed = set()

            for text in unique_texts:
                if text in processed:
                    continue

                canonical = text
                variants = [text]
                processed.add(text)

                emb1 = embedding_map[text]

                for other_text in unique_texts:
                    if other_text == text or other_text in processed:
                        continue

                    emb2 = embedding_map[other_text]

                    emb_sim = self._embedding_similarity(emb1, emb2)
                    fuzzy_sim = self._fuzzy_match(text, other_text)

                    if emb_sim >= self.similarity_threshold or fuzzy_sim >= self.fuzzy_threshold:
                        variants.append(other_text)
                        processed.add(other_text)

                canonical_to_variants[canonical] = variants

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
                    passage_ids=sorted(passage_ids),
                    embedding=embedding_map[canonical],
                )

            print(f"Reduced {len(unique_texts)} to {len(canonical_to_variants)} canonical entities")

        return canonical_map

    @staticmethod
    def remap_relationships(
        relationships: list[Relationship],
        canonical_map: dict[str, CanonicalEntity],
    ) -> list[Relationship]:
       
        variant_to_canonical = {}
        for canonical_entity in canonical_map.values():
            for variant in canonical_entity.variants:
                variant_to_canonical[variant] = canonical_entity.canonical_text

        remapped = []
        for rel in relationships:
            source_canon = variant_to_canonical.get(rel.source_entity, rel.source_entity)
            target_canon = variant_to_canonical.get(rel.target_entity, rel.target_entity)

            if source_canon in canonical_map and target_canon in canonical_map:
                remapped.append(
                    Relationship(
                        source_entity=source_canon,
                        relationship_type=rel.relationship_type,
                        target_entity=target_canon,
                        passage_id=rel.passage_id,
                        confidence=rel.confidence,
                    )
                )

        print(f"Remapped {len(relationships)} relationships, kept {len(remapped)}")
        return remapped


def save_canonical_entities(
    canonical_map: dict[str, CanonicalEntity],
    output_file: str,
):
   
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