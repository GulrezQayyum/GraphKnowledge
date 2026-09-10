import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.chunker import chunk_meditations
from src.deduplication import CanonicalEntity, EntityDeduplicator
from src.extraction import Entity, EntityRelationshipExtractor, Relationship
from src.graph_builder import KnowledgeGraph
from src.query_engine import QueryEngine


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        return np.array([
            [1.0, 0.0] if text == "reason" else [0.99, 0.01]
            for text in texts
        ])


def fake_client(content):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )


def graph_entities():
    return {
        "reason": CanonicalEntity("reason", "CONCEPT", passage_ids=["p1"]),
        "peace": CanonicalEntity("peace", "STATE", passage_ids=["p2"]),
    }


def test_chunker_keeps_short_numbered_sections():
    chunks = chunk_meditations("BOOK I.\n1. First.\n2. Second.")

    assert [chunk["passage_id"] for chunk in chunks] == ["bookI_1", "bookI_2"]
    assert chunks[0]["metadata"]["book_int"] == 1


def test_extraction_parses_groq_chat_response():
    content = json.dumps({
        "entities": [{"text": "reason", "type": "CONCEPT"}],
        "relationships": [{
            "source": "reason",
            "type": "leads_to",
            "target": "peace",
        }],
    })
    extractor = EntityRelationshipExtractor(client=fake_client(content))

    entities, relationships = extractor.extract("text", "p1")

    assert entities == [Entity("reason", "CONCEPT", "p1")]
    assert relationships == [Relationship("reason", "leads_to", "peace", "p1")]
    assert extractor.failures == []


def test_extraction_reports_invalid_response():
    extractor = EntityRelationshipExtractor(
        client=fake_client('{"entities":[{"text":"reason","type":"INVALID"}]}')
    )

    entities, relationships = extractor.extract("text", "p1")

    assert entities == []
    assert relationships == []
    assert extractor.failures[0].passage_id == "p1"


def test_deduplication_is_deterministic_and_does_not_mutate_relationships():
    entities = [
        Entity("reasons", "CONCEPT", "p2"),
        Entity("reason", "CONCEPT", "p1"),
    ]
    relationship = Relationship("reasons", "leads_to", "reason", "p2")
    deduplicator = EntityDeduplicator(model=FakeEmbeddingModel())

    canonical = deduplicator.deduplicate(entities)
    remapped = EntityDeduplicator.remap_relationships([relationship], canonical)

    assert list(canonical) == ["reason"]
    assert remapped[0].source_entity == "reason"
    assert relationship.source_entity == "reasons"


def test_graph_persistence_preserves_parallel_relationships(tmp_path: Path):
    relationships = [
        Relationship("reason", "relates_to", "peace", "p1"),
        Relationship("reason", "leads_to", "peace", "p2"),
    ]
    graph = KnowledgeGraph()
    graph.build_from_canonical(graph_entities(), relationships)

    path = tmp_path / "graph.json"
    graph.save(str(path))
    loaded = KnowledgeGraph.load(str(path))

    assert loaded.graph.number_of_edges() == 2
    assert {
        edge[1] for edge in loaded.traverse("reason").edges_traversed
    } == {"relates_to", "leads_to"}


def test_query_engine_returns_connected_passages():
    graph = KnowledgeGraph()
    graph.build_from_canonical(
        graph_entities(),
        [Relationship("reason", "leads_to", "peace", "p1")],
    )
    engine = QueryEngine(graph, {"p1": "Reason leads to peace."})

    result = engine.query_entity("rea", max_hops=1)

    assert result.found_entities == ["reason"]
    assert result.retrieved_passages == ["p1", "p2"]
    assert engine.get_passages(["p1"]) == {"p1": "Reason leads to peace."}