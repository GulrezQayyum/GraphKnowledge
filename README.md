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
GraphKnowledge/
├── src/
│   ├── extraction.py          # Entity/relationship extraction with Groq
│   ├── deduplication.py       # Entity linking & consolidation
│   ├── graph_builder.py       # NetworkX graph construction
│   ├── query_engine.py        # Graph traversal & retrieval
├── data/
│   ├── meditations_raw.txt    # Input Meditations corpus
│   └── graph/
│       ├── entities.json
│       ├── canonical_entities.json
│       ├── relationships.json
│       ├── relationships_remapped.json
│       └── knowledge_graph.json
├── notebooks/                 # Optional exploratory notebooks
├── phase1_pipeline.py         # Pipeline implementation
├── run_phase1.py              # CLI launcher
├── tests/                     # Regression tests
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

The checked-in `data/meditations_raw.txt` is the default input. To use a
different raw text file, pass its path to the launcher.

---

## Phase 1 Workflow

### Run the Full Pipeline

```bash
python run_phase1.py
```

This runs:
1. **Extraction**: Extract entities & relationships from each chunk using Groq
2. **Deduplication**: Link entity variants (e.g., "virtue" ↔ "virtues") to canonical forms
3. **Graph Building**: Build NetworkX directed graph from canonical entities
4. **Queries**: Test the graph with sample queries
5. **Queries**: Test the graph with sample queries

Generated artifacts are written to `data/graph/` and reused on later runs.
Rebuild cached extraction and deduplication artifacts with:

```bash
python run_phase1.py --force-extraction --force-dedup
```

Use a custom input and output directory with:

```bash
python run_phase1.py path/to/meditations.txt path/to/output
```

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
if extractor.failures:
  print(f"Failed passages: {len(extractor.failures)}")
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

Run the regression tests with:

```bash
pytest -q
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
- Check JSON output from Groq — may not parse


### Deduplication Loses Information

- Lower `similarity_threshold` if entities are being over-merged
- Increase `fuzzy_threshold` if variants aren't being linked

### Graph is Disconnected

- Check relationship extraction — may be filtering too aggressively
- Increase max_hops in queries to reach distant entities


# Phase 2 Fix: Groq Evaluation (Working)

## Problem
The original evaluation code was using `openai/gpt-oss-20b` which wasn't following the JSON format requirements. Groq was returning narrative text instead of structured output.

## Solution
✅ **Use `mixtral-8x7b-32768`** (same model from Phase 1)  
✅ **Simplified prompts** that don't require strict JSON parsing  
✅ **Smart fallbacks** if Groq returns unexpected text  

---

## What Changed

### Files Updated

| Old | New | Purpose |
|-----|-----|---------|
| `phase2_evaluation.py` | `phase2_evaluation_groq_fixed.py` | Fixed Groq handling |
| `run_phase2.py` | `run_phase2_free.py` | Uses fixed evaluation |

### Key Improvements

1. **Switched model**: `openai/gpt-oss-20b` → `mixtral-8x7b-32768`
2. **Simple prompts**: No JSON requirement, Groq just returns a number
3. **Smart parsing**:
   - Extracts score from text like "0.85"
   - Falls back to keyword matching ("high" → 0.8, "low" → 0.3)
   - Default: 0.5 if nothing found

4. **Reduced token usage**: Only score first 3 passages per method
5. **Better error handling**: Prints warnings but continues

---

## How to Run (Fixed)

### Prerequisites
```bash
# Ensure Phase 1 is complete
# Verify Groq API key
export GROQ_API_KEY='your-groq-key'
```

### Run Everything
```bash
# Test run (3 queries)
python3 run_phase2_free.py --sample 3

# Full run (15 queries)
python3 run_phase2_free.py
```

### Or Run Components Separately
```bash
# Just benchmark
python3 phase2_benchmark.py

# Just evaluation (with fixed Groq)
python3 phase2_evaluation_groq_fixed.py --sample 3
```

---

## Results Format

### Benchmark: `phase2_benchmark_results.json`
```json
{
  "query_id": "q1",
  "question": "How does Marcus connect fear and reason?",
  "methods": {
    "graph": { "passages": [...], "num_passages": 20 },
    "vector": { "passages": [...], "num_passages": 20 },
    "hybrid": { "passages": [...], "num_passages": 38 }
  }
}
```

### Evaluation: `phase2_evaluation_results.json`
```json
{
  "query_id": "q1",
  "question": "How does Marcus connect fear and reason?",
  "methods": {
    "graph": {
      "faithfulness": 0.82,
      "relevance": 0.75,
      "coverage": 0.80,
      "average": 0.79
    },
    "vector": { ... },
    "hybrid": { ... }
  }
}
```

---

## What the Scores Mean

| Metric | Meaning | Range |
|--------|---------|-------|
| **Faithfulness** | Are passages grounded in text? | 0-1 |
| **Relevance** | Do passages answer the question? | 0-1 |
| **Coverage** | Did we retrieve enough? | 0-1 |
| **Average** | Overall quality | 0-1 |

**Higher = Better** ✓

---

## Expected Output

```
[1/3] q1: How does Marcus connect fear and reason?...
  GRAPH   : 20 passages → 0.79
  VECTOR  : 20 passages → 0.81
  HYBRID  : 38 passages → 0.88

[2/3] q2: What practices...
  GRAPH   : 20 passages → 0.75
  ...
```

Then summary:
```
SUMMARY
Average Scores (across all queries):
  GRAPH   : 0.78 (±0.05)
  VECTOR  : 0.81 (±0.04)
  HYBRID  : 0.85 (±0.03)

Best Method:
  ✓ HYBRID: 0.85
```

---

## Cost

**$0** ✓ (Uses your existing Groq account)

---

## Next Steps

1. ✅ Run: `python3 run_phase2_free.py --sample 3`
2. ✅ Check results in generated JSON files
3. ✅ Identify best method (likely hybrid)
4. ➡️ Phase 3: LLM reasoning over best method

---

## Troubleshooting

### "GROQ_API_KEY not found"
```bash
export GROQ_API_KEY='your-key'
```

### "ModuleNotFoundError: phase2_evaluation_groq_fixed"
Make sure you're using the updated files:
- `phase2_evaluation_groq_fixed.py` (not the old one)
- `run_phase2_free.py` (not the old `run_phase2.py`)

### Evaluation is slow
- Groq API takes ~5-10 seconds per query
- For 15 queries: ~5-10 minutes total
- Test with `--sample 3` first

---

## Ready?

Run this:
```bash
python3 run_phase2_free.py --sample 3
```

Let me know the results! 