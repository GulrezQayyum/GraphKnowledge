"""
Phase 3: Answer Generation using Groq

Takes retrieved passages + query + entity context
→ Generates coherent, grounded answers
"""

import json
import os
from typing import Optional
from dataclasses import dataclass

from groq import Groq


@dataclass
class GeneratedAnswer:
    """Generated answer with metadata."""
    query_id: str
    question: str
    answer: str
    passages_used: int
    method: str  # "graph", "vector", or "hybrid"
    model: str


class AnswerGenerator:
    """Generate answers using Groq."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client."""
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    def generate_answer(
        self,
        question: str,
        passages: list[str],
        context: Optional[str] = None,
    ) -> str:
        """
        Generate answer from retrieved passages.

        Args:
            question: The question to answer
            passages: Retrieved passage texts
            context: Optional extra context (entities, etc.)

        Returns:
            Generated answer
        """
        if not passages:
            return "No relevant passages found to answer this question."

        # Build context
        passage_text = "\n".join([f"[P{i+1}]\n{p}\n" for i, p in enumerate(passages[:5])])

        extra_context = f"\n\nContext information:\n{context}" if context else ""

        prompt = f"""You are answering based ONLY on the passages below about Marcus Aurelius's Meditations.

Question: {question}

Passages:
{passage_text}
{extra_context}

Generate a coherent, direct answer (2-3 sentences) grounded in the passages.
If passages don't address the question, say so clearly.

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                reasoning_effort="low",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"


def run_answer_generation(
    benchmark_results_file: str = "phase2_benchmark_results.json",
    passages_file: str = "data/graph/meditations_chunks.json",
    output_file: str = "phase3_answers.json",
    method: str = "hybrid",
    graph=None,
):
    """
    Generate answers for all evaluation queries.

    Args:
        benchmark_results_file: Path to benchmark results
        passages_file: Path to passages
        output_file: Where to save answers
        method: Which method to use ("graph", "vector", or "hybrid")
        graph: Optional KnowledgeGraph for entity context
    """
    print("=" * 70)
    print("PHASE 3: ANSWER GENERATION")
    print("=" * 70)

    # Load benchmark results
    with open(benchmark_results_file) as f:
        benchmark_results = json.load(f)

    # Load passages
    with open(passages_file) as f:
        chunks = json.load(f)
    passages_dict = {c["passage_id"]: c["text"] for c in chunks}

    print(f"\nLoaded {len(benchmark_results)} queries")
    print(f"Using {method.upper()} retrieval method\n")

    generator = AnswerGenerator()
    answers = []

    for i, result in enumerate(benchmark_results):
        query_id = result["query_id"]
        question = result["question"]

        print(f"[{i+1}/{len(benchmark_results)}] {query_id}: {question[:50]}...", flush=True)

        # Get passages for this method
        passage_ids = result["methods"][method]["passages"][:10]
        passage_texts = [passages_dict[pid] for pid in passage_ids if pid in passages_dict]

        # Get entity context if available
        entity_context = None
        if graph:
            entities = graph.search_entities(question)
            if entities:
                entity_context = f"Key entities in question: {', '.join(entities[:5])}"

        # Generate answer
        answer = generator.generate_answer(question, passage_texts, entity_context)

        answers.append({
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "passages_used": len(passage_texts),
            "method": method,
            "model": generator.model,
        })

        print(f"  ✓ Generated ({len(passage_texts)} passages)")

    # Save answers
    with open(output_file, "w") as f:
        json.dump(answers, f, indent=2)

    print(f"\n✓ Answers saved to {output_file}")
    return answers


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.graph_builder import KnowledgeGraph

    kg = KnowledgeGraph.load("data/graph/knowledge_graph.json")
    answers = run_answer_generation(graph=kg)