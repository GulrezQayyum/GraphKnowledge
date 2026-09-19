"""
Phase 2: Benchmark Graph vs Vector vs Hybrid Retrieval

For each evaluation query:
1. Retrieve using GRAPH-ONLY (entity traversal)
2. Retrieve using VECTOR-ONLY (semantic similarity)
3. Retrieve using HYBRID (both methods)

Compare coverage, relevance, and efficiency.
"""

import json
import os
from typing import Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    method: str  # "graph", "vector", "hybrid"
    query: str
    passages: list[str]  # List of passage IDs
    scores: Optional[list[float]] = None  # Relevance scores (for vector/hybrid)
    num_passages: int = 0
    
    def __post_init__(self):
        self.num_passages = len(self.passages)


class HybridRetriever:
    """
    Combines graph-based and vector-based retrieval.
    """
    
    def __init__(self, graph, passages, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize hybrid retriever.
        
        Args:
            graph: KnowledgeGraph instance
            passages: Dict mapping {passage_id -> passage_text}
            embedding_model: Sentence transformer model
        """
        self.graph = graph
        self.passages = passages
        self.embedder = SentenceTransformer(embedding_model)
        
        # Pre-compute embeddings for all passages
        print("Computing passage embeddings...")
        passage_texts = [passages[pid] for pid in passages.keys()]
        self.passage_ids = list(passages.keys())
        self.passage_embeddings = self.embedder.encode(passage_texts, convert_to_numpy=True)
        print(f"Embedded {len(self.passage_ids)} passages")
    
    def retrieve_graph_only(
        self,
        query: str,
        max_hops: int = 2,
        top_k: Optional[int] = 20,
    ) -> RetrievalResult:
        """
        Retrieve using graph traversal only.
        
        Args:
            query: Query text (entity name to start from)
            max_hops: Maximum hops in graph
            top_k: Return top K passages (None = all)
            
        Returns:
            RetrievalResult with passages from graph traversal
        """
        # Find entities matching query
        found_entities = self.graph.search_entities_in_text(query)
        
        if not found_entities:
            return RetrievalResult(
                method="graph",
                query=query,
                passages=[],
            )
        
        # Traverse from each found entity
        passage_scores = {}
        for entity in found_entities:
            traversal = self.graph.traverse(entity, max_hops=max_hops, direction="both")
            for passage_id in traversal.passages_reached:
                passage_scores[passage_id] = passage_scores.get(passage_id, 0) + len(
                    traversal.traversed_entities
                )

            # Passages that explicitly mention a query entity are more useful
            # than passages reached only through a neighboring graph node.
            for passage_id in self.graph.entity_to_passages.get(entity, []):
                passage_scores[passage_id] = passage_scores.get(passage_id, 0) + 2
        
        ranked_passages = sorted(
            passage_scores,
            key=lambda passage_id: (-passage_scores[passage_id], passage_id),
        )
        passages = ranked_passages if top_k is None else ranked_passages[:top_k]
        
        return RetrievalResult(
            method="graph",
            query=query,
            passages=passages,
        )
    
    def retrieve_vector_only(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.3,
    ) -> RetrievalResult:
        """
        Retrieve using semantic similarity only.
        
        Args:
            query: Query text
            top_k: Return top K passages
            threshold: Minimum similarity score
            
        Returns:
            RetrievalResult with passages ranked by similarity
        """
        # Embed query
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        
        # Compute similarities
        similarities = np.dot(self.passage_embeddings, query_embedding) / (
            np.linalg.norm(self.passage_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Filter by threshold and get top-k
        valid_indices = np.where(similarities >= threshold)[0]
        valid_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        valid_indices = valid_indices[:top_k]
        
        passages = [self.passage_ids[i] for i in valid_indices]
        scores = similarities[valid_indices].tolist()
        
        return RetrievalResult(
            method="vector",
            query=query,
            passages=passages,
            scores=scores,
        )
    
    def retrieve_hybrid(
        self,
        query: str,
        max_hops: int = 2,
        vector_top_k: int = 20,
        alpha: float = 0.5,
    ) -> RetrievalResult:
        """
        Retrieve using both graph and vector methods, then combine.
        
        Args:
            query: Query text
            max_hops: Graph traversal hops
            vector_top_k: Vector retrieval top-k
            alpha: Weight for combining (0.5 = equal weight)
            
        Returns:
            RetrievalResult combining both methods
        """
        # Get graph results
        graph_result = self.retrieve_graph_only(
            query, max_hops=max_hops, top_k=vector_top_k
        )
        
        # Get vector results
        vector_result = self.retrieve_vector_only(query, top_k=vector_top_k)
        
        # Combine: union of passages, score by presence in both
        graph_passages = set(graph_result.passages)
        vector_passages = set(vector_result.passages)
        
        # Passages in both methods get higher score
        combined_set = graph_passages | vector_passages
        
        combined_passages = []
        combined_scores = []
        
        for pid in combined_set:
            in_graph = pid in graph_passages
            in_vector = pid in vector_passages
            
            # Score: 1.0 if in both, 0.5 if in one
            score = (in_graph * alpha) + (in_vector * (1 - alpha))
            combined_passages.append((pid, score))
        
        # Sort by score
        combined_passages.sort(key=lambda x: (-x[1], x[0]))
        combined_passages = combined_passages[:vector_top_k]
        passages = [p[0] for p in combined_passages]
        scores = [p[1] for p in combined_passages]
        
        return RetrievalResult(
            method="hybrid",
            query=query,
            passages=passages,
            scores=scores,
        )


def run_benchmark(
    graph,
    passages,
    evaluation_queries_file: str = "evaluation_queries.json",
    output_file: str = "phase2_benchmark_results.json",
):
    """
    Run complete benchmark comparing all retrieval methods.
    
    Args:
        graph: KnowledgeGraph instance
        passages: Dict of passages
        evaluation_queries_file: Path to queries JSON
        output_file: Where to save results
    """
    print("=" * 70)
    print("PHASE 2: RETRIEVAL BENCHMARK")
    print("=" * 70)
    
    # Load evaluation queries
    with open(evaluation_queries_file) as f:
        eval_queries = json.load(f)
    
    print(f"\nLoaded {len(eval_queries)} evaluation queries\n")
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(graph, passages)
    
    # Run benchmark
    results = []
    
    for i, query_data in enumerate(eval_queries):
        query_id = query_data["id"]
        question = query_data["question"]
        difficulty = query_data["difficulty"]
        
        print(f"[{i+1}/{len(eval_queries)}] {query_id}: {question[:60]}...")
        
        # Retrieve using all three methods
        graph_result = retriever.retrieve_graph_only(question)
        vector_result = retriever.retrieve_vector_only(question, top_k=20)
        hybrid_result = retriever.retrieve_hybrid(question)
        
        # Store results
        results.append({
            "query_id": query_id,
            "question": question,
            "difficulty": difficulty,
            "expected_entities": query_data.get("expected_entities", []),
            "methods": {
                "graph": {
                    "passages": graph_result.passages,
                    "num_passages": graph_result.num_passages,
                },
                "vector": {
                    "passages": vector_result.passages,
                    "scores": vector_result.scores,
                    "num_passages": vector_result.num_passages,
                },
                "hybrid": {
                    "passages": hybrid_result.passages,
                    "scores": hybrid_result.scores,
                    "num_passages": hybrid_result.num_passages,
                },
            },
        })
        
        print(f"  Graph: {graph_result.num_passages} passages")
        print(f"  Vector: {vector_result.num_passages} passages")
        print(f"  Hybrid: {hybrid_result.num_passages} passages")
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Benchmark complete. Results saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    graph_counts = [r["methods"]["graph"]["num_passages"] for r in results]
    vector_counts = [r["methods"]["vector"]["num_passages"] for r in results]
    hybrid_counts = [r["methods"]["hybrid"]["num_passages"] for r in results]
    
    print(f"\nGraph-Only:")
    print(f"  Avg passages: {np.mean(graph_counts):.1f}")
    print(f"  Min: {np.min(graph_counts)}, Max: {np.max(graph_counts)}")
    
    print(f"\nVector-Only:")
    print(f"  Avg passages: {np.mean(vector_counts):.1f}")
    print(f"  Min: {np.min(vector_counts)}, Max: {np.max(vector_counts)}")
    
    print(f"\nHybrid:")
    print(f"  Avg passages: {np.mean(hybrid_counts):.1f}")
    print(f"  Min: {np.min(hybrid_counts)}, Max: {np.max(hybrid_counts)}")
    
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    from src.graph_builder import KnowledgeGraph
    
    # Load graph
    print("Loading graph...")
    kg = KnowledgeGraph.load("data/graph/knowledge_graph.json")
    
    # Load passages
    print("Loading passages...")
    with open("data/graph/meditations_chunks.json") as f:
        chunks = json.load(f)
    passages = {c["passage_id"]: c["text"] for c in chunks}
    
    # Run benchmark
    results = run_benchmark(kg, passages)