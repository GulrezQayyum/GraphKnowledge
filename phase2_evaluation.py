"""
Phase 2: Evaluation using Groq (FIXED - Simple, Robust)

Uses a Groq chat-completions model for lightweight judging.
Simple prompts that don't require strict JSON parsing.
"""

import json
import os
import re
import statistics
from typing import Optional

from groq import Groq


class GroqEvaluator:
    """Simple evaluation using Groq."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client."""
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-20b")

    def _complete(self, prompt: str) -> str:
        """Run a short evaluator prompt through Groq's chat API."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=50,
            reasoning_effort="low",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def extract_score(self, text: str) -> float:
        """Extract score (0-1) from response text."""
        # Look for patterns like "0.85" or "score: 0.85"
        matches = re.findall(r"(?:score[:\s]+)?(\d+\.?\d*)", text.lower())
        if matches:
            score = float(matches[-1])
            if score > 1:
                score = score / 100  # Convert if given as 85 instead of 0.85
            return min(1.0, max(0.0, score))
        
        # Fallback: Look for text indicators
        if any(word in text.lower() for word in ["high", "very", "excellent", "strong"]):
            return 0.8
        if any(word in text.lower() for word in ["moderate", "some", "partial"]):
            return 0.6
        if any(word in text.lower() for word in ["low", "weak", "poor", "not"]):
            return 0.3
        
        return 0.5  # Default

    def score_faithfulness(self, question: str, passages: list[str]) -> float:
        """Score: Are passages faithful and grounded?"""
        if not passages:
            return 0.0

        # Take first 3 passages to avoid token limit
        passages = passages[:3]
        context = "\n".join([f"P{i+1}: {p[:150]}" for i, p in enumerate(passages)])

        prompt = f"""Question: {question}

Passages:
{context}

Are these passages grounded in the text and faithful to the question?
Respond with just a score from 0 to 1, where:
- 1 = Passages are very faithful and grounded
- 0.5 = Passages are somewhat related
- 0 = Passages are not faithful

Score: """

        try:
            text = self._complete(prompt)
            return self.extract_score(text)
        except Exception as e:
            print(f"    ⚠️  Faithfulness error: {e}")
            return 0.5

    def score_relevance(self, question: str, passages: list[str]) -> float:
        """Score: Are passages relevant to the question?"""
        if not passages:
            return 0.0

        # Take first 3 passages
        passages = passages[:3]
        context = "\n".join([f"P{i+1}: {p[:150]}" for i, p in enumerate(passages)])

        prompt = f"""Question: {question}

Passages:
{context}

Do these passages help answer the question?
Respond with just a score from 0 to 1, where:
- 1 = Passages directly answer the question
- 0.5 = Passages are somewhat relevant
- 0 = Passages don't answer the question

Score: """

        try:
            text = self._complete(prompt)
            return self.extract_score(text)
        except Exception as e:
            print(f"    ⚠️  Relevance error: {e}")
            return 0.5

    def score_coverage(self, num_passages: int, expected: int = 10) -> float:
        """Score: Coverage heuristic (passages / expected)."""
        if num_passages == 0:
            return 0.0
        if num_passages >= expected:
            return 1.0
        return num_passages / expected


def run_evaluation(
    benchmark_results_file: str = "phase2_benchmark_results.json",
    passages_file: str = "data/graph/meditations_chunks.json",
    output_file: str = "phase2_evaluation_results.json",
    sample_queries: Optional[int] = None,
):
    """
    Run evaluation on benchmark results using Groq (FREE).
    """
    print("=" * 70)
    print("PHASE 2: EVALUATION (Groq - openai/gpt-oss-20b)")
    print("=" * 70)

    # Load benchmark results
    with open(benchmark_results_file) as f:
        benchmark_results = json.load(f)

    if sample_queries:
        benchmark_results = benchmark_results[:sample_queries]
        print(f"\n(Evaluating first {sample_queries} queries)\n")

    # Load passages
    with open(passages_file) as f:
        chunks = json.load(f)
    passages_dict = {c["passage_id"]: c["text"] for c in chunks}

    print(f"Loaded {len(benchmark_results)} queries")
    print(f"Loaded {len(passages_dict)} passages\n")

    # Initialize evaluator
    evaluator = GroqEvaluator()

    # Evaluate
    evaluation_results = []

    for i, result in enumerate(benchmark_results):
        query_id = result["query_id"]
        question = result["question"]

        print(f"[{i+1}/{len(benchmark_results)}] {query_id}: {question[:50]}...", flush=True)

        query_evals = {
            "query_id": query_id,
            "question": question,
            "methods": {},
        }

        # Evaluate each method
        for method in ["graph", "vector", "hybrid"]:
            passage_ids = result["methods"][method]["passages"][:10]  # Use max 10
            passage_texts = [
                passages_dict[pid] for pid in passage_ids if pid in passages_dict
            ]

            num_passages = len(passage_texts)
            print(f"  {method.upper():8s}: {num_passages:2d} passages → ", end="", flush=True)

            # Score (with simple fallbacks)
            faith = evaluator.score_faithfulness(question, passage_texts)
            relevance = evaluator.score_relevance(question, passage_texts)
            coverage = evaluator.score_coverage(num_passages)

            average = (faith + relevance + coverage) / 3

            query_evals["methods"][method] = {
                "method": method,
                "num_passages": num_passages,
                "faithfulness": round(faith, 3),
                "relevance": round(relevance, 3),
                "coverage": round(coverage, 3),
                "average": round(average, 3),
            }

            print(f"{average:.3f}")

        evaluation_results.append(query_evals)

    # Save results
    with open(output_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\n✓ Evaluation complete. Saved to {output_file}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    methods_scores = {"graph": [], "vector": [], "hybrid": []}

    for result in evaluation_results:
        for method in ["graph", "vector", "hybrid"]:
            if method in result["methods"]:
                avg = result["methods"][method].get("average", 0)
                methods_scores[method].append(avg)

    print("\nAverage Scores (across all queries):")
    for method in ["graph", "vector", "hybrid"]:
        scores = methods_scores[method]
        if scores:
            mean = statistics.mean(scores)
            stdev = statistics.stdev(scores) if len(scores) > 1 else 0
            print(f"  {method.upper():8s}: {mean:.3f} (±{stdev:.3f})")

    print("\nBest Method:")
    best_method = max(
        methods_scores,
        key=lambda m: statistics.mean(methods_scores[m]) if methods_scores[m] else 0,
    )
    best_score = statistics.mean(methods_scores[best_method])
    print(f"  ✓ {best_method.upper()}: {best_score:.3f}")

    return evaluation_results


if __name__ == "__main__":
    results = run_evaluation(sample_queries=3)