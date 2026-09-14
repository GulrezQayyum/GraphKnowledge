#!/usr/bin/env python3
"""
Phase 2 Orchestrator: Run benchmark and evaluation end-to-end.

Usage:
  python3 run_phase2.py                    # Full run (all 15 queries)
  python3 run_phase2.py --sample 3         # Test run (3 queries)
  python3 run_phase2.py --benchmark-only   # Skip evaluation
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, ".")

from src.graph_builder import KnowledgeGraph
from phase2_benchmark import run_benchmark
from phase2_evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 benchmark and evaluation")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run only on first N queries (for testing)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run benchmark only, skip evaluation",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only, skip benchmark",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GraphKnowledge Phase 2: Multi-Hop Evaluation & Benchmarking")
    print("=" * 80)
    
    # Check prerequisites
    print("\n[1/3] Checking prerequisites...")
    
    required_files = [
        "data/graph/knowledge_graph.json",
        "data/graph/meditations_chunks.json",
        "evaluation_queries.json",
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        print("   Make sure Phase 1 is complete!")
        sys.exit(1)
    
    print("✓ All prerequisites found")
    
    # Load graph
    print("\n[2/3] Loading graph...")
    kg = KnowledgeGraph.load("data/graph/knowledge_graph.json")
    print(f"✓ Graph loaded: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
    
    # Load passages
    print("      Loading passages...")
    with open("data/graph/meditations_chunks.json") as f:
        chunks = json.load(f)
    passages = {c["passage_id"]: c["text"] for c in chunks}
    print(f"✓ Passages loaded: {len(passages)} chunks")
    
    # Run benchmark
    if not args.eval_only:
        print("\n[3/3] Running benchmark...")
        print("      (Comparing graph-only vs vector-only vs hybrid retrieval)\n")
        
        try:
            benchmark_results = run_benchmark(
                kg,
                passages,
                evaluation_queries_file="evaluation_queries.json",
                output_file="phase2_benchmark_results.json",
            )
            print("\n✓ Benchmark complete!")
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Run evaluation
    if not args.benchmark_only:
        print("\n[4/4] Running RAGAS evaluation...")
        print("      (Evaluating faithfulness, relevance, context recall)\n")
        
        try:
            sample_queries = args.sample
            eval_results = run_evaluation(
                benchmark_results_file="phase2_benchmark_results.json",
                passages_file="data/graph/meditations_chunks.json",
                output_file="phase2_evaluation_results.json",
                sample_queries=sample_queries,
            )
            print("\n✓ Evaluation complete!")
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    
    print("\nGenerated files:")
    print("  ✓ phase2_benchmark_results.json  — Retrieval comparison")
    if not args.benchmark_only:
        print("  ✓ phase2_evaluation_results.json — RAGAS scores")
    
    print("\nNext steps:")
    print("  1. Review results in the generated JSON files")
    print("  2. Identify the best retrieval method (likely hybrid)")
    print("  3. Move to Phase 3: LLM reasoning over retrieved passages")
    
    print("\nFor detailed analysis, see PHASE2_README.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()