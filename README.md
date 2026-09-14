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


# Phase 2: Multi-Hop Evaluation & Benchmarking

## Overview

Phase 2 compares three retrieval methods:

1. **Graph-Only**: Entity traversal (1-2 hops)
2. **Vector-Only**: Semantic similarity search
3. **Hybrid**: Combines both methods

## Files

### 1. `evaluation_queries.json`
15 hand-crafted multi-hop questions grounded in Meditations:
- "How does Marcus connect fear and reason?"
- "What practices lead to virtue?"
- "How does discipline lead to tranquility?"
- etc.

Each query includes:
- `question`: The query text
- `difficulty`: easy/medium/hard
- `expected_entities`: Entities that should appear
- `expected_relationship_chain`: The path through the graph

### 2. `phase2_benchmark.py`
Runs retrieval benchmark comparing all three methods.

**What it does:**
1. For each evaluation query
2. Retrieve using graph-only (traverse relationships)
3. Retrieve using vector-only (semantic similarity)
4. Retrieve using hybrid (combine results)
5. Save results to `phase2_benchmark_results.json`

**Outputs:**
- Passages retrieved by each method
- Number of passages per method
- Comparison of coverage

### 3. `phase2_evaluation.py`
Evaluates retrieval quality using RAGAS metrics.

**Metrics:**
- **Faithfulness**: Are retrieved passages faithful to the question?
- **Relevance**: Do passages answer the question?
- **Context Recall**: Did we retrieve the right passages?

**Outputs:**
- RAGAS scores for each method
- Average scores across all queries
- Best-performing method

## Setup

### Prerequisites

1. **Phase 1 complete** ✅
   - `data/graph/knowledge_graph.json` exists
   - `data/graph/meditations_chunks.json` exists

2. **Dependencies**
```bash
pip install --break-system-packages ragas langchain-anthropic
```

3. **API Keys**
```bash
export GROQ_API_KEY='your-groq-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
```

## Running Phase 2

### Step 1: Run Benchmark

```bash
python3 phase2_benchmark.py
```

This generates `phase2_benchmark_results.json`:
```json
[
  {
    "query_id": "q1",
    "question": "How does Marcus connect fear and reason?",
    "difficulty": "medium",
    "methods": {
      "graph": {
        "passages": ["bookI_3", "bookI_7", ...],
        "num_passages": 12
      },
      "vector": {
        "passages": ["bookI_5", "bookII_1", ...],
        "scores": [0.85, 0.78, ...],
        "num_passages": 15
      },
      "hybrid": {
        "passages": ["bookI_3", "bookI_5", ...],
        "num_passages": 20
      }
    }
  },
  ...
]
```

### Step 2: Run Evaluation

```bash
python3 phase2_evaluation.py
```

This generates `phase2_evaluation_results.json`:
```json
[
  {
    "query_id": "q1",
    "question": "How does Marcus connect fear and reason?",
    "methods": {
      "graph": {
        "faithfulness": 0.82,
        "relevance": 0.75,
        "context_recall": 0.80,
        "average": 0.79
      },
      "vector": {
        "faithfulness": 0.71,
        "relevance": 0.88,
        "context_recall": 0.85,
        "average": 0.81
      },
      "hybrid": {
        "faithfulness": 0.85,
        "relevance": 0.90,
        "context_recall": 0.88,
        "average": 0.88
      }
    }
  }
]
```

## Understanding Results

### Benchmark Results

Look at `phase2_benchmark_results.json`:

**Question**: "How does Marcus connect fear and reason?"

```
Graph:   12 passages (entity traversal: fear → reason)
Vector:  15 passages (semantic similarity to question)
Hybrid:  20 passages (union of both + ranking)
```

**What to look for:**
- Does graph return fewer but more focused passages?
- Does vector return broader coverage?
- Does hybrid balance both?

### Evaluation Results

Look at `phase2_evaluation_results.json`:

**Best method** wins on average RAGAS score:

```
Graph:   0.79 average
Vector:  0.81 average
Hybrid:  0.88 average ✓ BEST
```

**Interpretation:**
- **Faithfulness**: Passages stick to facts in Meditations
- **Relevance**: Passages actually answer the question
- **Context Recall**: We found the right passages

## Expected Outcomes

### If Graph Wins:
- Entity relationships are precise
- Multi-hop traversal captures intent well
- Conservative but high-quality retrieval

### If Vector Wins:
- Semantic embeddings are good for this corpus
- Graph might be missing implicit relationships
- Consider improving entity extraction

### If Hybrid Wins:
- Both methods complement each other ✓
- Combining them captures more nuance
- This is the expected outcome for balanced systems

## Troubleshooting

### "No passages retrieved for graph"
- Query entity might not exist in graph
- Check: `kg.search_entities("query_word")`
- May need to adjust entity extraction or queries

### RAGAS evaluation is slow
- It's using Claude to evaluate each passage
- Start with `sample_queries=3` (in `phase2_evaluation.py`)
- Full evaluation is OK for 15 queries

### High variance in RAGAS scores
- Some questions are harder than others
- Some retrieval results are genuinely better
- Look at difficulty level to contextualize

## Next Steps

Once Phase 2 is done:

1. **Analyze results**: Which method wins? Why?
2. **Refine if needed**: 
   - If graph underperforms: Improve entity extraction
   - If vector underperforms: Better embeddings
   - If hybrid wins: Use it for Phase 3
3. **Move to Phase 3**: LLM reasoning over best method

## Files to Submit

After Phase 2 is complete, share:

1. ✅ `phase2_benchmark_results.json` — Retrieval comparison
2. ✅ `phase2_evaluation_results.json` — RAGAS scores
3. ✅ Summary of findings (which method won?)

---

**Ready to start?**

```bash
python3 phase2_benchmark.py
python3 phase2_evaluation.py
```

Let me know the results! 🚀