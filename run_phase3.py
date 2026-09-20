#!/usr/bin/env python3
"""
Phase 3 Orchestrator: Answer Generation & Evaluation

Complete capstone pipeline:
1. Generate answers using hybrid retrieval
2. Evaluate answers with RAGAS
3. Generate HTML report
4. Summary & insights
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, ".")

from src.graph_builder import KnowledgeGraph
from answer_generator import run_answer_generation
from phase3_evaluation import run_evaluation
from report_generator import generate_html_report


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Answer Generation & Evaluation (Capstone)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["graph", "vector", "hybrid"],
        help="Retrieval method to use",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GraphKnowledge Phase 3: Answer Generation & Evaluation (CAPSTONE)")
    print("=" * 80)

    # Check prerequisites
    print("\n[1/5] Checking prerequisites...")
    required = [
        "data/graph/knowledge_graph.json",
        "data/graph/meditations_chunks.json",
        "phase2_benchmark_results.json",
        "evaluation_queries.json",
    ]

    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print(f" Missing: {', '.join(missing)}")
        print("   Complete Phase 1 & 2 first!")
        sys.exit(1)

    print("✓ All prerequisites ready")

    # Load graph
    print("\n[2/5] Loading knowledge graph...")
    kg = KnowledgeGraph.load("data/graph/knowledge_graph.json")
    print(f"✓ {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")

    # Generate answers
    print(f"\n[3/5] Generating answers using {args.method.upper()} retrieval...")
    try:
        answers = run_answer_generation(graph=kg, method=args.method)
        print("✓ Answers generated")
    except Exception as e:
        print(f" Answer generation failed: {e}")
        sys.exit(1)

    # Evaluate answers
    print("\n[4/5] Evaluating answers with RAGAS...")
    try:
        evaluations = run_evaluation()
        print("✓ Evaluation complete")
    except Exception as e:
        print(f" Evaluation failed: {e}")
        sys.exit(1)

    # Generate report
    if not args.no_report:
        print("\n[5/5] Generating HTML report...")
        try:
            generate_html_report()
            print("✓ Report generated")
        except Exception as e:
            print(f"⚠️  Report generation failed: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE - CAPSTONE FINISHED")
    print("=" * 80)

    print("\n RESULTS SUMMARY:")
    print(f"  • Answers generated: {len(answers)}")
    print(f"  • Evaluations completed: {len(evaluations)}")

    # Calculate averages
    avg_faith = sum(e["metrics"]["faithfulness"] for e in evaluations) / len(evaluations)
    avg_relevance = sum(e["metrics"]["relevance"] for e in evaluations) / len(evaluations)
    avg_recall = sum(e["metrics"]["context_recall"] for e in evaluations) / len(evaluations)
    overall_avg = sum(e["metrics"]["average"] for e in evaluations) / len(evaluations)

    print(f"\n QUALITY METRICS:")
    print(f"  • Faithfulness:   {avg_faith:.3f}")
    print(f"  • Relevance:      {avg_relevance:.3f}")
    print(f"  • Context Recall: {avg_recall:.3f}")
    print(f"  • OVERALL SCORE:  {overall_avg:.3f}")

    print(f"\n GENERATED FILES:")
    print(f"  ✓ phase3_answers.json")
    print(f"  ✓ phase3_evaluation.json")
    if not args.no_report:
        print(f"  ✓ phase3_report.html (open in browser)")

    print(f"\n🎓 PROJECT COMPLETE!")
    print(f"  Phase 1: Entity Extraction ✓")
    print(f"  Phase 2: Multi-hop Evaluation ✓")
    print(f"  Phase 3: Answer Generation ✓")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()