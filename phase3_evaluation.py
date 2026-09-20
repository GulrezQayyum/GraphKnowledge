"""
Phase 3: Evaluate Generated Answers using RAGAS

Metrics:
- Faithfulness: Is answer grounded in passages?
- Relevance: Does answer address the question?
- Context Recall: Did we use relevant passages?
"""

import json
import os
import re
import statistics
from typing import Optional

from groq import Groq


class AnswerEvaluator:
    """Evaluate answers using Groq."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client."""
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = "mixtral-8x7b-32768"

    def extract_score(self, text: str) -> float:
        """Extract score from response."""
        matches = re.findall(r"(\d+\.?\d*)", text.lower())
        if matches:
            score = float(matches[-1])
            if score > 1:
                score = score / 100
            return min(1.0, max(0.0, score))

        if any(w in text.lower() for w in ["high", "very", "excellent", "strong"]):
            return 0.8
        if any(w in text.lower() for w in ["moderate", "some", "partial"]):
            return 0.6
        if any(w in text.lower() for w in ["low", "weak", "poor", "not"]):
            return 0.3

        return 0.5

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        passages: list[str],
    ) -> float:
        """Score: Is answer faithful to passages?"""
        context = "\n".join([f"P{i+1}: {p[:100]}" for i, p in enumerate(passages[:3])])

        prompt = f"""Question: {question}

Generated Answer: {answer}

Passages (source):
{context}

Is the answer faithful and grounded in the passages?
Score 0-1 where 1 = very faithful, 0 = not faithful.

Score: """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            return self.extract_score(response.content[0].text)
        except:
            return 0.5

    def evaluate_relevance(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Score: Does answer address the question?"""
        prompt = f"""Question: {question}

Answer: {answer}

Does the answer directly address the question?
Score 0-1 where 1 = directly answers, 0 = doesn't answer.

Score: """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            return self.extract_score(response.content[0].text)
        except:
            return 0.5

    def evaluate_context_recall(
        self,
        question: str,
        passages: list[str],
    ) -> float:
        """Score: Were passages relevant?"""
        context = "\n".join([f"P{i+1}: {p[:100]}" for i, p in enumerate(passages[:3])])

        prompt = f"""Question: {question}

Passages:
{context}

Are these passages relevant to answer the question?
Score 0-1 where 1 = very relevant, 0 = not relevant.

Score: """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            return self.extract_score(response.content[0].text)
        except:
            return 0.5


def run_evaluation(
    answers_file: str = "phase3_answers.json",
    benchmark_results_file: str = "phase2_benchmark_results.json",
    passages_file: str = "data/graph/meditations_chunks.json",
    output_file: str = "phase3_evaluation.json",
):
    """Evaluate all generated answers."""
    print("=" * 70)
    print("PHASE 3: ANSWER EVALUATION (RAGAS)")
    print("=" * 70)

    # Load answers
    with open(answers_file) as f:
        answers = json.load(f)

    # Load benchmark results (for passage IDs)
    with open(benchmark_results_file) as f:
        benchmark_results = json.load(f)

    # Load passages
    with open(passages_file) as f:
        chunks = json.load(f)
    passages_dict = {c["passage_id"]: c["text"] for c in chunks}

    print(f"\nLoaded {len(answers)} answers")
    print(f"Loaded {len(passages_dict)} passages\n")

    evaluator = AnswerEvaluator()
    evaluations = []

    for i, answer_data in enumerate(answers):
        query_id = answer_data["query_id"]
        question = answer_data["question"]
        answer = answer_data["answer"]

        print(f"[{i+1}/{len(answers)}] {query_id}: {question[:50]}...", flush=True)

        # Get passages used
        benchmark = next(b for b in benchmark_results if b["query_id"] == query_id)
        method = answer_data["method"]
        passage_ids = benchmark["methods"][method]["passages"][:10]
        passage_texts = [passages_dict[pid] for pid in passage_ids if pid in passages_dict]

        # Evaluate
        faith = evaluator.evaluate_faithfulness(question, answer, passage_texts)
        relevance = evaluator.evaluate_relevance(question, answer)
        recall = evaluator.evaluate_context_recall(question, passage_texts)

        average = (faith + relevance + recall) / 3

        evaluations.append({
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "metrics": {
                "faithfulness": round(faith, 3),
                "relevance": round(relevance, 3),
                "context_recall": round(recall, 3),
                "average": round(average, 3),
            },
        })

        print(f"  Faithfulness: {faith:.3f}, Relevance: {relevance:.3f}, Avg: {average:.3f}")

    # Save evaluations
    with open(output_file, "w") as f:
        json.dump(evaluations, f, indent=2)

    print(f"\n✓ Evaluation complete. Saved to {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    faith_scores = [e["metrics"]["faithfulness"] for e in evaluations]
    relevance_scores = [e["metrics"]["relevance"] for e in evaluations]
    recall_scores = [e["metrics"]["context_recall"] for e in evaluations]
    avg_scores = [e["metrics"]["average"] for e in evaluations]

    print(f"\nFaithfulness:   {statistics.mean(faith_scores):.3f} (±{statistics.stdev(faith_scores):.3f})")
    print(f"Relevance:      {statistics.mean(relevance_scores):.3f} (±{statistics.stdev(relevance_scores):.3f})")
    print(f"Context Recall: {statistics.mean(recall_scores):.3f} (±{statistics.stdev(recall_scores):.3f})")
    print(f"\nOVERALL AVERAGE: {statistics.mean(avg_scores):.3f} (±{statistics.stdev(avg_scores):.3f})")

    return evaluations


if __name__ == "__main__":
    evaluations = run_evaluation()