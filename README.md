# GraphKnowledge

GraphKnowledge is a multi-phase advanced RAG project built around Marcus Aurelius's *Meditations*. It extracts a knowledge graph from the text, compares graph, vector, and hybrid retrieval, and generates evidence-grounded answers with claim-level evaluation.

The project demonstrates a complete retrieval-augmented generation workflow:

1. Extract entities and relationships from a philosophical corpus.
2. Deduplicate entities and build a queryable NetworkX graph.
3. Benchmark graph, vector, and hybrid retrieval.
4. Generate answers from retrieved passages.
5. Evaluate claims against cited evidence and produce an HTML report.

## Current Results

The checked-in Phase 3 artifacts were produced with hybrid retrieval over 15 evaluation queries:

| Metric | Result |
| --- | ---: |
| Claim grounding | 0.867 |
| Citation coverage | 0.867 |
| Citation precision | 0.867 |
| Overall answer score | 0.867 |
| Retrieval coverage | 1.000 |
| Abstention accuracy | 1.000* |
| Evaluator validity | 1.000 |

\* Abstention accuracy is currently based on one explicitly labeled abstention query, q14. Treat it as an initial signal rather than a broad reliability estimate.

The evaluator completed all 15 judgments successfully: 13 evidence-bearing answers received claims and two queries received valid zero-claim judgments. The earlier evaluator truncation problem was addressed with structured JSON output, a larger response budget, and an eight-claim limit.

## Project Structure

```text
GraphKnowledge/
├── src/
│   ├── chunker.py              # Split the corpus into numbered passages
│   ├── extraction.py           # Extract entities and relationships with Groq
│   ├── deduplication.py        # Link entity variants to canonical entities
│   ├── graph_builder.py        # Build and persist the NetworkX graph
│   └── query_engine.py         # Entity search, traversal, and passage lookup
├── data/
│   ├── meditations_raw.txt     # Input corpus
│   └── graph/                  # Phase 1 graph and passage artifacts
├── evaluation_queries.json     # 15 multi-hop benchmark questions
├── phase1_pipeline.py         # Phase 1 implementation
├── run_phase1.py              # Phase 1 CLI
├── phase2_benchmark.py        # Graph/vector/hybrid retrieval benchmark
├── phase2_evaluation.py       # Groq retrieval scoring
├── run_phase2.py              # Phase 2 CLI
├── answer_generator.py        # Grounded answer generation
├── phase3_evaluation.py       # Claim-level evidence evaluation
├── report_generator.py        # HTML report generation
├── run_phase3.py              # Phase 3 CLI
├── tests/                     # Regression tests
├── phase2_benchmark_results.json
├── phase2_evaluation_results.json
├── phase3_answers.json
├── phase3_evaluation.json
├── phase3_report.html
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.12 or a compatible recent Python version. A virtual environment is recommended:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set the Groq API key before running the LLM-backed phases:

```bash
export GROQ_API_KEY="your-groq-api-key"
```

Optional model overrides are supported through:

```bash
export GROQ_MODEL="openai/gpt-oss-20b"
export GROQ_EVAL_MODEL="openai/gpt-oss-20b"
```

Do not edit files inside `venv/`; project configuration belongs in the source files or environment variables.

## Phase 1: Build the Knowledge Graph

Run the full extraction, deduplication, graph-building, and sample-query pipeline:

```bash
python3 run_phase1.py
```

Optional arguments:

```bash
python3 run_phase1.py --force-extraction --force-dedup
python3 run_phase1.py path/to/meditations.txt path/to/output-directory
```

Phase 1 writes graph artifacts to `data/graph/`, including:

- `meditations_chunks.json`
- `entities.json`
- `relationships.json`
- `canonical_entities.json`
- `relationships_remapped.json`
- `knowledge_graph.json`

The graph stores canonical entities, typed relationships, passage IDs, and metadata needed for multi-hop retrieval.

## Phase 2: Benchmark Retrieval

Phase 2 compares three retrieval methods over the questions in `evaluation_queries.json`:

- **Graph:** entity matching and multi-hop graph traversal.
- **Vector:** sentence-transformer semantic similarity.
- **Hybrid:** graph scores, vector scores, and lexical overlap combined into a ranked result.

Run the full benchmark and Groq-based retrieval evaluation:

```bash
python3 run_phase2.py
```

For a smaller evaluation sample:

```bash
python3 run_phase2.py --sample 3
```

The main outputs are:

- `phase2_benchmark_results.json`: ranked passage IDs for each method.
- `phase2_evaluation_results.json`: faithfulness, relevance, coverage, and average retrieval scores.

Run only the benchmark implementation directly when needed:

```bash
python3 phase2_benchmark.py
```

## Phase 3: Generate and Evaluate Answers

Run the capstone pipeline with hybrid retrieval:

```bash
python3 run_phase3.py
```

Choose another retrieval method or skip report generation:

```bash
python3 run_phase3.py --method graph
python3 run_phase3.py --method vector
python3 run_phase3.py --method hybrid --no-report
```

Phase 3 performs five steps:

1. Check Phase 1 and Phase 2 prerequisites.
2. Load the knowledge graph.
3. Generate answers from the selected method's top 10 passages.
4. Extract atomic claims and judge each claim against the retrieved passages.
5. Generate `phase3_report.html`.

The generated answer records contain the query, answer, retrieval method, passage count, and model. The evaluation records contain:

- `claim_grounding`: proportion of claims supported by the passages.
- `citation_coverage`: proportion of claims with evidence IDs.
- `citation_precision`: proportion of supported claims that have evidence IDs.
- `retrieval_coverage`: proportion of evaluator evidence IDs present in the retrieved passages. Abstentions without evidence are reported as `N/A`.
- `abstention_accuracy`: correctness for queries with an explicit abstention label.
- `evaluator_valid`: whether the evaluator returned valid claims or a valid zero-claim judgment.

Evaluator diagnostics also record response status, response length, finish reason, parse errors, and a bounded response preview. This distinguishes a genuine zero-claim judgment from JSON truncation or an API failure.

You can run evaluation and report generation separately:

```bash
python3 phase3_evaluation.py
python3 -c "from report_generator import generate_html_report; generate_html_report()"
```

Open `phase3_report.html` in a browser to inspect aggregate and per-query results.

## Testing

Run the regression suite from the project root:

```bash
pytest -q
```

The tests cover chunking, extraction parsing, entity deduplication, graph persistence, graph traversal, and natural-language entity matching.

## Core Graph Concepts

The extraction pipeline represents concepts such as virtue, duty, and reason; people such as Marcus and Epictetus; practices such as discipline; and states such as fear, anger, and tranquility.

Relationships include:

| Relationship | Example | Meaning |
| --- | --- | --- |
| `relates_to` | virtue -> duty | General conceptual connection |
| `leads_to` | discipline -> tranquility | Practice or cause leading to an outcome |
| `teaches` | Epictetus -> acceptance | Instruction or influence |
| `resolved_by` | fear -> reason | Problem and response |
| `opposes` | emotion -> virtue | Tension or contrast |
| `requires` | virtue -> discipline | Prerequisite relationship |
| `embodies` | courage -> virtue | Specific expression of a concept |

For example, a query about fear and reason can use graph traversal to collect passages connected to both concepts before the answer generator synthesizes a response.

## Troubleshooting

### Missing API key

Verify that `GROQ_API_KEY` is exported in the same shell used to run the command:

```bash
echo "$GROQ_API_KEY"
```

### Rate limits

Groq rate limits can cause evaluator requests to return `429`. The pipeline records these as `api_error` and excludes invalid evaluator results from quality averages. Wait for the limit to reset before rerunning evaluation.

### Evaluator parse errors

The evaluator requests JSON output and records `finish_reason`. A response with `finish_reason=length` is truncated and marked `parse_error`. The current implementation uses a 1200-token output budget and limits the evaluator to eight concise claims.

### Stale artifacts

Phase 3 reads the existing Phase 2 benchmark results. Re-run Phase 2 after changing retrieval code, then rerun Phase 3:

```bash
python3 run_phase2.py
python3 run_phase3.py
```

## Project Status

Phase 1, Phase 2, and Phase 3 are implemented. The current checked-in report is the final hybrid-retrieval baseline. Future evaluation work should add more explicitly labeled answerable and unanswerable queries so abstention accuracy is measured over a broader set.
