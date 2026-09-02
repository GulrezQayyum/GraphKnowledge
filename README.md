# GraphKnowledge — Phase 1: Entity Extraction & Graph Building

Building a knowledge graph from *Meditations* by Marcus Aurelius to enable multi-hop reasoning and entity-based retrieval.

**Part of Advanced RAG Engineering** — Learn-by-building project series.

---

## Overview

### What is GraphKnowledge?

GraphKnowledge extracts entities (concepts, people, practices, emotions) and their relationships from philosophical texts, then builds a queryable knowledge graph. Unlike pure semantic search, it enables **multi-hop reasoning**: answering questions that require connecting ideas scattered across different passages.

**Example**:
- Query: *"How does Marcus connect fear to reason?"*
- Traditional RAG: Might miss the connection
- GraphKnowledge: Traverses `fear → relates_to → reason`, collects all connected passages, synthesizes answer

### Real-World Reference

See [Graphify](https://graphify.net/) — an open-source implementation for code understanding that builds queryable knowledge graphs from codebases, docs, and diagrams. Same pattern, different domain.

---

## Project Structure

```
graphknowledge/
├── src/
│   ├── extraction.py          # Entity/relationship extraction with Groq
│   ├── deduplication.py       # Entity linking & consolidation
│   ├── graph_builder.py       # NetworkX graph construction
│   ├── query_engine.py        # Graph traversal & retrieval
│   └── utils.py               # Helpers (optional)
├── data/
│   ├── chunks.json            # Meditations corpus (from ChunkLab)
│   └── graph/
│       ├── entities.json
│       ├── canonical_entities.json
│       ├── relationships.json
│       ├── relationships_remapped.json
│       └── knowledge_graph.json
├── notebooks/
│   ├── 01_exploration.ipynb   # Understand extraction quality
│   └── 02_evaluation.ipynb    # Test graph queries
├── phase1_pipeline.py         # Main orchestrator
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from [https://console.groq.com](https://console.groq.com).

### 3. Prepare Corpus

You need a chunked Meditations corpus from ChunkLab. Place it at `data/chunks.json`.

Expected format (list of chunks):
```json
[
  {
    "text": "Book 1, Chapter 2: On losing Marcus's father...",
    "metadata": { "book": 1, "chapter": 2 }
  },
  ...
]
```

Or if you have a different format, modify `load_meditations_corpus()` in `phase1_pipeline.py`.

---

## Phase 1 Workflow

### Run the Full Pipeline

```bash
python phase1_pipeline.py data/chunks.json data/graph
```

This runs:
1. **Extraction**: Extract entities & relationships from each chunk using Groq
2. **Deduplication**: Link entity variants (e.g., "virtue" ↔ "virtues") to canonical forms
3. **Graph Building**: Build NetworkX directed graph from canonical entities
4. **Queries**: Test the graph with sample queries
5. **Interactive Session**: Try your own queries

### Pipeline Steps (Detailed)

#### Step 1: Entity & Relationship Extraction

Uses Groq to extract structured knowledge:
- **Entities**: CONCEPT (virtue, fear), PERSON (Epictetus), PRACTICE (meditation), STATE (tranquility)
- **Relationships**: relates_to, leads_to, teaches, resolved_by, opposes, requires, embodies

Output: `entities.json`, `relationships.json`

```python
from src.extraction import EntityRelationshipExtractor

extractor = EntityRelationshipExtractor()
entities, relationships = extractor.extract_batch(passages)
```

#### Step 2: Entity Deduplication

Consolidates entity mentions using embedding similarity + fuzzy matching:
- "virtue", "virtues" → canonical "virtue"
- "fear", "fears" → canonical "fear"
- Handles typos and variations

Output: `canonical_entities.json`

```python
from src.deduplication import EntityDeduplicator

deduplicator = EntityDeduplicator(similarity_threshold=0.85)
canonical_map = deduplicator.deduplicate(entities)
```

#### Step 3: Graph Building

Constructs NetworkX directed graph:
- Nodes: Canonical entities
- Edges: Relationships with type labels
- Metadata: Passage IDs, passage counts per entity

Output: `knowledge_graph.json`

```python
from src.graph_builder import KnowledgeGraph

kg = KnowledgeGraph()
kg.build_from_canonical(canonical_map, remapped_relationships)
```

#### Step 4: Querying

Search entities and traverse the graph:

```python
from src.query_engine import QueryEngine

query_engine = QueryEngine(kg, passages)
result = query_engine.query_entity("fear", max_hops=2)
# Returns: found entities, traversal paths, retrieved passages
```

---

## Core Concepts

### Entity Types

| Type | Examples | Used For |
|------|----------|----------|
| **CONCEPT** | virtue, duty, reason, desire | Core philosophical ideas |
| **PERSON** | Marcus, Epictetus, Socrates | Historical/philosophical figures |
| **PRACTICE** | meditation, discipline, reflection | Actions & habits |
| **STATE** | fear, anger, grief, tranquility | Emotions & mental states |

### Relationship Types

| Type | Example | Meaning |
|------|---------|---------|
| **relates_to** | virtue ↔ duty | Concepts are connected |
| **leads_to** | discipline → tranquility | Cause/effect or practice/outcome |
| **teaches** | Epictetus → acceptance | Person teaches concept |
| **resolved_by** | fear → reason | Problem/solution |
| **opposes** | virtue ↔ vice | Contrasts |
| **requires** | virtue ← discipline | Prerequisite |
| **embodies** | courage ← virtue | Specific instance of general concept |

### Graph Traversal

Given query entity, traverse N hops to find connected entities:

```
Start: "fear"
  ↓
Hop 1: fear → [relates_to → reason, resolved_by → acceptance, opposes → courage]
  ↓
Hop 2: reason → [relates_to → virtue], acceptance → [leads_to → peace], ...
```

Collect all passages tagged with each reached entity → retrieve for LLM synthesis.

---

## API Reference

### EntityRelationshipExtractor

```python
extractor = EntityRelationshipExtractor(api_key=None)
entities, relationships = extractor.extract(passage, passage_id)
entities, relationships = extractor.extract_batch(passages_dict)
```

### EntityDeduplicator

```python
deduplicator = EntityDeduplicator(similarity_threshold=0.85, fuzzy_threshold=0.80)
canonical_map = deduplicator.deduplicate(entities)
remapped = deduplicator.remap_relationships(relationships, canonical_map)
```

### KnowledgeGraph

```python
kg = KnowledgeGraph()
kg.build_from_canonical(canonical_map, relationships)

result = kg.traverse(start_entity="fear", max_hops=2, direction="both")
entity_info = kg.get_entity_info("virtue")
matches = kg.search_entities("vir")  # Partial match

kg.save("knowledge_graph.json")
kg = KnowledgeGraph.load("knowledge_graph.json")
stats = kg.stats()
```

### QueryEngine

```python
query_engine = QueryEngine(kg, passages_dict)
result = query_engine.query_entity("fear", max_hops=2)
passages = query_engine.get_passages(result.retrieved_passages)
formatted = query_engine.format_result(result, show_passages=True)
query_engine.interactive_session()  # Start interactive CLI
```

---

## Success Metrics (Phase 1)

- [ ] **Extraction**: 100+ unique entities extracted from corpus
- [ ] **Deduplication**: Entity count reduced by 30-40% via linking
- [ ] **Graph**: 200+ relationships; graph has interesting structure (not disconnected)
- [ ] **Queries**: Multi-hop traversal works (e.g., "fear" → "reason" → retrieve passages)
- [ ] **Corpus Coverage**: 80%+ of passages have at least 1 entity

---

## Next Steps (Phase 2)

Once Phase 1 is solid:
1. **Multi-hop Retrieval**: Compare graph-only vs. vector-only vs. hybrid retrieval
2. **Benchmark**: Create 15-20 multi-hop evaluation queries
3. **Metrics**: Use RAGAS (faithfulness, context recall) to measure quality

---

## Troubleshooting

### Extraction Fails or Returns Empty

- Check Groq API key is valid
- Verify passage length (very short passages may fail)

### Deduplication Loses Information

- Lower `similarity_threshold` if entities are being over-merged
- Increase `fuzzy_threshold` if variants aren't being linked

### Graph is Disconnected

- Check relationship extraction — may be filtering too aggressively
- Increase max_hops in queries to reach distant entities

---

## Questions?

See project notes in `/areas/graphknowledge.md` or refer to [Graphify](https://graphify.net/) for real-world patterns.
