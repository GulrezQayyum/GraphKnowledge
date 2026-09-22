"""Phase 3: Claim-level evaluation of generated answers."""

import json
import os
import statistics
from typing import Optional

from groq import Groq


class AnswerEvaluator:
    """Evaluate answers using Groq."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client."""
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-20b")

    def evaluate_claims(
        self,
        question: str,
        answer: str,
        passages: dict[str, str],
    ) -> list[dict]:
        """Extract atomic claims and verify each against cited evidence."""
        context = "\n\n".join(f"[{pid}]\n{text}" for pid, text in passages.items())

        prompt = f"""Question: {question}

Generated Answer: {answer}

Passages (source):
{context}

Split the answer into atomic factual claims. For each claim, decide whether it is
fully supported by the passages. Return JSON only in this exact shape:
{{"claims": [{{"claim": "...", "supported": true, "evidence_ids": ["bookI_3"]}}]}}
Use only exact passage IDs shown above. A claim is unsupported if it adds an
interpretation, detail, or attribution not present in the passages. Do not merge
multiple independently verifiable facts into one claim.

JSON: """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                reasoning_effort="low",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            claims = json.loads(content).get("claims", [])
            valid_ids = set(passages)
            return [
                {
                    "claim": str(item.get("claim", "")),
                    "supported": bool(item.get("supported", False)),
                    "evidence_ids": [
                        evidence_id
                        for evidence_id in item.get("evidence_ids", [])
                        if evidence_id in valid_ids
                    ],
                }
                for item in claims
                if item.get("claim")
            ]
        except (json.JSONDecodeError, AttributeError, TypeError, IndexError):
            return []
        except Exception as error:
            print(f"  Evaluation unavailable: {error}")
            return []


def run_evaluation(
    answers_file: str = "phase3_answers.json",
    benchmark_results_file: str = "phase2_benchmark_results.json",
    passages_file: str = "data/graph/meditations_chunks.json",
    output_file: str = "phase3_evaluation.json",
):
    """Evaluate all generated answers."""
    print("=" * 70)
    print("PHASE 3: ANSWER EVALUATION (CLAIM LEVEL)")
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
        passage_texts = {
            pid: passages_dict[pid] for pid in passage_ids if pid in passages_dict
        }

        claims = evaluator.evaluate_claims(question, answer, passage_texts)
        claim_count = len(claims)
        supported_count = sum(claim["supported"] for claim in claims)
        cited_count = sum(bool(claim["evidence_ids"]) for claim in claims)
        grounding = supported_count / claim_count if claim_count else 0.0
        citation_coverage = cited_count / claim_count if claim_count else 0.0
        citation_precision = (
            sum(
                bool(claim["evidence_ids"])
                for claim in claims
                if claim["supported"]
            ) / supported_count
            if supported_count
            else 0.0
        )

        average = (grounding + citation_coverage + citation_precision) / 3

        evaluations.append({
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "metrics": {
                "claim_grounding": round(grounding, 3),
                "citation_coverage": round(citation_coverage, 3),
                "citation_precision": round(citation_precision, 3),
                "average": round(average, 3),
            },
            "claims": claims,
        })

        print(
            f"  Claims: {claim_count}, Grounding: {grounding:.3f}, "
            f"Citation coverage: {citation_coverage:.3f}, Avg: {average:.3f}"
        )

    # Save evaluations
    with open(output_file, "w") as f:
        json.dump(evaluations, f, indent=2)

    print(f"\n✓ Evaluation complete. Saved to {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    grounding_scores = [e["metrics"]["claim_grounding"] for e in evaluations]
    coverage_scores = [e["metrics"]["citation_coverage"] for e in evaluations]
    precision_scores = [e["metrics"]["citation_precision"] for e in evaluations]
    avg_scores = [e["metrics"]["average"] for e in evaluations]

    print(f"\nClaim Grounding:    {statistics.mean(grounding_scores):.3f} (±{statistics.stdev(grounding_scores):.3f})")
    print(f"Citation Coverage:  {statistics.mean(coverage_scores):.3f} (±{statistics.stdev(coverage_scores):.3f})")
    print(f"Citation Precision: {statistics.mean(precision_scores):.3f} (±{statistics.stdev(precision_scores):.3f})")
    print(f"\nOVERALL AVERAGE: {statistics.mean(avg_scores):.3f} (±{statistics.stdev(avg_scores):.3f})")

    return evaluations


if __name__ == "__main__":
    evaluations = run_evaluation()