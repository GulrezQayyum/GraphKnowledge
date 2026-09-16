#!/usr/bin/env python3
"""
Phase 2 Orchestrator (FREE - Groq Only)

Usage:
  python3 run_phase2_free.py                # Full run (15 queries)
  python3 run_phase2_free.py --sample 3     # Test run (3 queries)
  python3 run_phase2_free.py --benchmark-only  # Skip evaluation
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, ".")

from src.graph_builder import KnowledgeGraph
from phase2_benchmark import run_benchmark
from phase2_evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Benchmark & Evaluate (FREE - Groq Only)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Test run: first N queries only",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip evaluation, benchmark only",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GraphKnowledge Phase 2: Benchmark & Evaluate (FREE - Groq)")
    print("=" * 80)
    
    # Check prerequisites
    print("\n[1/3] Checking prerequisites...")
    
    required = [
        "data/graph/knowledge_graph.json",
        "data/graph/meditations_chunks.json",
        "evaluation_queries.json",
    ]
    
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        print("   Complete Phase 1 first!")
        sys.exit(1)
    
    print("✓ All files ready")
    
    # Load graph
    print("\n[2/3] Loading graph...")
    kg = KnowledgeGraph.load("data/graph/knowledge_graph.json")
    print(f"✓ {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
    
    # Load passages
    with open("data/graph/meditations_chunks.json") as f:
        chunks = json.load(f)
    passages = {c["passage_id"]: c["text"] for c in chunks}
    print(f"✓ {len(passages)} passages")
    
    # Run benchmark
    if not args.benchmark_only:
        print("\n[3/3] Running benchmark...")
        try:
            run_benchmark(kg, passages)
            print("✓ Benchmark done")
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            sys.exit(1)
    
    # Run evaluation
    print("\n[4/4] Running evaluation (Groq - FREE)...")
    try:
        run_evaluation(
            sample_queries=args.sample,
        )
        print("✓ Evaluation done")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE (Cost: $0)")
    print("=" * 80)
    print("\nResults:")
    print("  ✓ phase2_benchmark_results.json")
    print("  ✓ phase2_evaluation_results.json")
    print("\nBest method analysis:")
    print("  → Check 'average' scores in evaluation results")
    print("  → Hybrid typically wins")
    print("\nNext: Phase 3 (LLM reasoning)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()