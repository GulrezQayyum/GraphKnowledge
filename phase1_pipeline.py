"""
Phase 1 Pipeline: Entity Extraction, Deduplication, and Graph Building.

Orchestrates the full Phase 1 workflow:
1. Save raw Meditations text
2. Chunk Meditations by book + section
3. Extract entities and relationships using Groq
4. Deduplicate entities
5. Build knowledge graph
6. Save artifacts
"""

import json
import os
# Local imports
from src.chunker import chunk_meditations, save_chunks
from src.extraction import EntityRelationshipExtractor, save_extractions, load_extractions
from src.deduplication import EntityDeduplicator, save_canonical_entities, load_canonical_entities
from src.graph_builder import KnowledgeGraph
from src.query_engine import QueryEngine


def save_raw_meditations(meditations_text: str, output_file: str):
    """
    Save raw Meditations text to file.
    
    Args:
        meditations_text: Full Meditations text
        output_file: Path to save
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(meditations_text)
    print(f"Saved raw Meditations to {output_file}")


def load_meditations_corpus(chunks: list[dict]) -> dict[str, str]:
    """
    Convert chunks list to passages dict.
    
    Args:
        chunks: List of {passage_id, text, metadata}
        
    Returns:
        Dict mapping {passage_id -> passage_text}
    """
    passages = {chunk["passage_id"]: chunk["text"] for chunk in chunks}
    print(f"Loaded {len(passages)} passages from chunks")
    return passages


def run_phase1(
    meditations_text: str,
    output_dir: str = "data/graph",
    force_extraction: bool = False,
    force_dedup: bool = False,
):
    """
    Run Phase 1 pipeline from raw Meditations text.
    
    Args:
        meditations_text: Raw Meditations text
        output_dir: Output directory for artifacts
        force_extraction: Force re-extraction even if cache exists
        force_dedup: Force re-deduplication even if cache exists
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: ENTITY EXTRACTION & GRAPH BUILDING")
    print("=" * 60)

    # Step 0a: Save raw text
    print("\n[Step 0a] Saving raw Meditations...")
    raw_file = os.path.join(output_dir, "meditations_raw.txt")
    save_raw_meditations(meditations_text, raw_file)
    
    # Step 0b: Chunk corpus
    print("\n[Step 0b] Chunking Meditations by book + section...")
    chunks = chunk_meditations(meditations_text)
    
    chunks_file = os.path.join(output_dir, "meditations_chunks.json")
    save_chunks(chunks, chunks_file)
    
    # Step 0c: Load corpus
    print("\n[Step 0c] Converting chunks to passage dict...")
    passages = load_meditations_corpus(chunks)

    # Step 1: Extract entities and relationships
    print("\n[Step 1] Extracting entities and relationships...")
    entities_file = os.path.join(output_dir, "entities.json")
    relationships_file = os.path.join(output_dir, "relationships.json")

    if os.path.exists(entities_file) and os.path.exists(relationships_file) and not force_extraction:
        print("Loading cached extractions...")
        entities, relationships = load_extractions(entities_file, relationships_file)
    else:
        extractor = EntityRelationshipExtractor()
        entities, relationships = extractor.extract_batch(passages)
        save_extractions(entities, relationships, entities_file, relationships_file)

    print(f"Extracted {len(entities)} entity mentions from {len(passages)} passages")
    print(f"Extracted {len(relationships)} relationship mentions")

    # Step 2: Deduplicate entities
    print("\n[Step 2] Deduplicating entities...")
    canonical_entities_file = os.path.join(output_dir, "canonical_entities.json")
    if os.path.exists(canonical_entities_file) and not force_dedup and not force_extraction:
        print("Loading cached canonical entities...")
        canonical_map = load_canonical_entities(canonical_entities_file)
        deduplicator = None
    else:
        deduplicator = EntityDeduplicator(
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=0.85,
            fuzzy_threshold=0.80,
        )
        canonical_map = deduplicator.deduplicate(entities)
        save_canonical_entities(canonical_map, canonical_entities_file)

    # Print entity type breakdown
    print("\nCanonical Entities by Type:")
    by_type = {}
    for entity in canonical_map.values():
        if entity.entity_type not in by_type:
            by_type[entity.entity_type] = 0
        by_type[entity.entity_type] += 1
    for entity_type, count in by_type.items():
        print(f"  {entity_type}: {count}")

    # Step 3: Remap relationships to canonical entities
    print("\n[Step 3] Remapping relationships to canonical entities...")
    remapped_relationships = EntityDeduplicator.remap_relationships(relationships, canonical_map)

    remapped_relationships_file = os.path.join(output_dir, "relationships_remapped.json")
    rel_data = [
        {
            "source": r.source_entity,
            "type": r.relationship_type,
            "target": r.target_entity,
            "passage_id": r.passage_id,
            "confidence": r.confidence,
        }
        for r in remapped_relationships
    ]
    with open(remapped_relationships_file, "w") as f:
        json.dump(rel_data, f, indent=2)
    print(f"Saved {len(remapped_relationships)} remapped relationships")

    # Step 4: Build knowledge graph
    print("\n[Step 4] Building knowledge graph...")
    kg = KnowledgeGraph()
    kg.build_from_canonical(canonical_map, remapped_relationships)

    graph_file = os.path.join(output_dir, "knowledge_graph.json")
    kg.save(graph_file)

    # Print graph statistics
    print("\nGraph Statistics:")
    stats = kg.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Step 5: Test queries
    print("\n[Step 5] Testing graph queries...")
    query_engine = QueryEngine(kg, passages)

    test_queries = [
        "virtue",
        "fear",
        "Marcus",
        "reason",
        "death",
    ]

    print("\nSample Queries:")
    for query in test_queries:
        result = query_engine.query_entity(query, max_hops=2)
        print(f"\n  Query: '{query}'")
        print(f"    Found entities: {result.found_entities}")
        print(f"    Passages reached: {len(result.retrieved_passages)}")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print(f"Artifacts saved to {output_dir}/")
    print("=" * 60)

    return kg, passages, query_engine


if __name__ == "__main__":
    import sys

    # Usage: python phase1_pipeline.py [meditations_raw_file] [output_dir]
    meditations_file = sys.argv[1] if len(sys.argv) > 1 else "data/meditations_raw.txt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/graph"

    # Read raw Meditations
    if not os.path.exists(meditations_file):
        print(f"Error: Meditations file not found at {meditations_file}")
        sys.exit(1)

    with open(meditations_file, 'r') as f:
        meditations_text = f.read()

    kg, passages, query_engine = run_phase1(meditations_text, output_dir)

    # Start interactive session
    print("\nStarting interactive query session...\n")
    query_engine.interactive_session()