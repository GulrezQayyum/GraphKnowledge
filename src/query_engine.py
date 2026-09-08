"""
Query Engine for GraphKnowledge.

Handles user queries, entity extraction, graph traversal, and passage retrieval.
"""

from dataclasses import dataclass
from typing import Optional

from graph_builder import KnowledgeGraph, TraversalResult


@dataclass
class QueryResult:
    """Result of a query."""
    query_text: str
    found_entities: list[str]
    traversal_results: list[TraversalResult]
    retrieved_passages: list[str]
    explanation: str


class QueryEngine:
    """
    Query interface for GraphKnowledge.
    """

    def __init__(self, graph: KnowledgeGraph, passages: dict[str, str]):
        """
        Initialize query engine.
        
        Args:
            graph: KnowledgeGraph instance
            passages: Dict mapping {passage_id -> passage_text}
        """
        self.graph = graph
        self.passages = passages

    def query_entity(
        self,
        entity_name: str,
        max_hops: int = 2,
        direction: str = "both",
    ) -> QueryResult:
        """
        Query the graph starting from an entity.
        
        Args:
            entity_name: Entity to search for (can be partial)
            max_hops: Maximum traversal hops
            direction: "forward", "backward", or "both"
            
        Returns:
            QueryResult with traversed entities and passages
        """
        # Search for entities
        found_entities = self.graph.search_entities(entity_name)

        if not found_entities:
            return QueryResult(
                query_text=entity_name,
                found_entities=[],
                traversal_results=[],
                retrieved_passages=[],
                explanation=f"No entities found matching '{entity_name}'",
            )

        # Traverse from each found entity
        traversals = []
        all_passages = set()

        for entity in found_entities:
            traversal = self.graph.traverse(entity, max_hops=max_hops, direction=direction)
            traversals.append(traversal)
            all_passages.update(traversal.passages_reached)

        return QueryResult(
            query_text=entity_name,
            found_entities=found_entities,
            traversal_results=traversals,
            retrieved_passages=list(all_passages),
            explanation=f"Found {len(found_entities)} entities, traversed to {len(all_passages)} passages",
        )

    def get_passages(self, passage_ids: list[str]) -> dict[str, str]:
        """
        Retrieve passage texts by IDs.
        
        Args:
            passage_ids: List of passage IDs
            
        Returns:
            Dict mapping {passage_id -> passage_text}
        """
        return {pid: self.passages[pid] for pid in passage_ids if pid in self.passages}

    def format_result(self, result: QueryResult, show_passages: bool = True) -> str:
        """
        Format query result as readable text.
        
        Args:
            result: QueryResult
            show_passages: Whether to include passage text
            
        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"Query: {result.query_text}")
        lines.append(f"Status: {result.explanation}")
        lines.append("")

        if result.found_entities:
            lines.append("Found Entities:")
            for entity in result.found_entities:
                info = self.graph.get_entity_info(entity)
                lines.append(f"  • {entity} ({info['type']}, {info['passage_count']} passages)")
            lines.append("")

        if result.traversal_results:
            for trav in result.traversal_results:
                lines.append(f"Starting from: {trav.start_entity}")
                lines.append(f"  Traversed: {len(trav.traversed_entities)} entities")
                if trav.edges_traversed:
                    lines.append("  Relationships:")
                    for source, rel_type, target in trav.edges_traversed[:5]:  # Show first 5
                        lines.append(f"    {source} --({rel_type})--> {target}")
                    if len(trav.edges_traversed) > 5:
                        lines.append(f"    ... and {len(trav.edges_traversed) - 5} more")
                lines.append("")

        if show_passages and result.retrieved_passages:
            lines.append(f"Retrieved {len(result.retrieved_passages)} passages:")
            for passage_id in result.retrieved_passages[:3]:  # Show first 3
                if passage_id in self.passages:
                    passage_text = self.passages[passage_id]
                    preview = passage_text[:150] + "..." if len(passage_text) > 150 else passage_text
                    lines.append(f"  [{passage_id}]: {preview}")
            if len(result.retrieved_passages) > 3:
                lines.append(f"  ... and {len(result.retrieved_passages) - 3} more passages")

        return "\n".join(lines)

    def interactive_session(self):
        """
        Start an interactive query session.
        """
        print("GraphKnowledge Query Engine")
        print("Type entity names to query (e.g., 'virtue', 'fear', 'Marcus')")
        print("Type 'exit' to quit\n")

        while True:
            query = input("Query> ").strip()

            if query.lower() == "exit":
                break

            result = self.query_entity(query, max_hops=2, direction="both")
            print(self.format_result(result, show_passages=True))
            print()