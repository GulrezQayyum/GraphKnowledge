"""
Phase 2: RAGAS-Lite Evaluation using Groq (FREE)

Simplified evaluation using Groq instead of paid APIs:
- Faithfulness: Groq scores if context supports question
- Relevance: Groq scores if passages answer question
- Coverage: Simple heuristic (passage count / expected)

No external paid APIs needed!
"""

import json
import os
import re
from typing import Optional
from dataclasses import dataclass

from groq import Groq


@dataclass
class EvaluationMetrics:
    """Metrics for a single retrieval."""
    method: str
    query_id: str
    faithfulness_score: float
    relevance_score: float
    coverage_score: float
    
    @property
    def average(self) -> float:
        """Average of all metrics."""
        return (self.faithfulness_score + self.relevance_score + self.coverage_score) / 3


class GroqEvaluator:
    """
    Simple evaluation using Groq (free).
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client."""
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    
    def score_faithfulness(self, question: str, passages: list[str]) -> float:
        """
        Score 0-1: Are passages faithful to the question?
        Uses Groq to evaluate.
        """
        if not passages:
            return 0.0
        
        context = "\n".join([f"[P{i+1}] {p[:200]}" for i, p in enumerate(passages)])
        
        prompt = f"""Rate faithfulness (0-1) of these passages to the question.
Are they grounded in the text and not making things up?

Question: {question}

Passages:
{context}

Return ONLY valid JSON in this exact format: {{"score": 0.85}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_score(response, "faithfulness")
        except Exception as exc:
            raise RuntimeError(
                f"Groq faithfulness scoring failed for question {question!r}: {exc}"
            ) from exc
    
    def score_relevance(self, question: str, passages: list[str]) -> float:
        """
        Score 0-1: Are passages relevant to the question?
        Uses Groq to evaluate.
        """
        if not passages:
            return 0.0
        
        context = "\n".join([f"[P{i+1}] {p[:200]}" for i, p in enumerate(passages)])
        
        prompt = f"""Rate relevance (0-1) of these passages to the question.
Do they help answer it?

Question: {question}

Passages:
{context}

Return ONLY valid JSON in this exact format: {{"score": 0.82}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_score(response, "relevance")
        except Exception as exc:
            raise RuntimeError(
                f"Groq relevance scoring failed for question {question!r}: {exc}"
            ) from exc

    @staticmethod
    def _parse_score(response, metric: str) -> float:
        """Parse the numeric score returned by Groq's chat-completions API."""
        message = response.choices[0].message
        response_parts = [
            getattr(message, "content", "") or "",
            getattr(message, "reasoning", "") or "",
            getattr(message, "reasoning_content", "") or "",
        ]
        score_text = "\n".join(response_parts).strip()
        try:
            parsed = json.loads(score_text)
            score = float(parsed["score"])
            if 0.0 <= score <= 1.0:
                return score
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        matches = re.findall(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", score_text)
        if not matches:
            raise ValueError(f"Groq returned an invalid {metric} score: {score_text!r}")
        return min(1.0, max(0.0, float(matches[-1])))
    
    def score_coverage(self, num_passages: int, expected: int = 10) -> float:
        """
        Score 0-1: Coverage heuristic.
        Assumes ~10 passages is good coverage.
        """
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
    top_k: int = 20,
):
    """
    Run evaluation on benchmark results using Groq (FREE).
    
    Args:
        benchmark_results_file: Path to benchmark results
        passages_file: Path to passages
        output_file: Where to save evaluation
        sample_queries: Limit queries (for testing)
        top_k: Maximum passages used for LLM precision/relevance scoring
    """
    print("=" * 70)
    print("PHASE 2: EVALUATION (Using Groq - FREE)")
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
        
        print(f"[{i+1}/{len(benchmark_results)}] {query_id}: {question[:50]}...")
        
        query_evals = {
            "query_id": query_id,
            "question": question,
            "methods": {},
        }
        
        # Evaluate each method
        for method in ["graph", "vector", "hybrid"]:
            passage_ids = result["methods"][method]["passages"]
            passage_ids = passage_ids[:top_k]
            passage_texts = [passages_dict[pid] for pid in passage_ids if pid in passages_dict]
            
            num_passages = len(passage_texts)
            print(f"  {method.upper()}: {num_passages} passages → ", end="", flush=True)
            
            # Score
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
    
    print(f"\n✓ Evaluation complete. Results saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    import statistics
    
    methods_scores = {"graph": [], "vector": [], "hybrid": []}
    
    for result in evaluation_results:
        for method in ["graph", "vector", "hybrid"]:
            if method in result["methods"]:
                avg = result["methods"][method].get("average", 0)
                methods_scores[method].append(avg)
    
    print("\nAverage Scores:")
    for method in ["graph", "vector", "hybrid"]:
        scores = methods_scores[method]
        if scores:
            mean = statistics.mean(scores)
            stdev = statistics.stdev(scores) if len(scores) > 1 else 0
            print(f"  {method.upper():8s}: {mean:.3f} (±{stdev:.3f})")
    
    print("\nBest Method:")
    best_method = max(
        methods_scores,
        key=lambda m: statistics.mean(methods_scores[m]) if methods_scores[m] else 0
    )
    best_score = statistics.mean(methods_scores[best_method])
    print(f"  {best_method.upper()}: {best_score:.3f}")
    
    return evaluation_results


if __name__ == "__main__":
    # Run evaluation
    results = run_evaluation(sample_queries=3)  # Start with 3