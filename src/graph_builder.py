"""
Knowledge Graph Construction using NetworkX.

Builds a directed graph from canonical entities and relationships.
Provides graph persistence (JSON export) and basic traversal utilities.
"""

import json
import os
from typing import Optional
from dataclasses import dataclass

import networkx as nx

from deduplication import CanonicalEntity
from extraction import Relationship


@dataclass
class TraversalResult:
    """Result of graph traversal."""
    start_entity: str
    traversed_entities: list[str]
    edges_traversed: list[tuple[str, str, str]]  # (source, relationship_type, target)
    passages_reached: list[str]  # passage_ids connected to traversed entities


class KnowledgeGraph:
    """
    Manages a knowledge graph of Meditations concepts and relationships.
    """

    def __init__(self):
        """Initialize an empty directed graph."""
        self.graph = nx.DiGraph()
        self.entity_to_passages = {}  # {canonical_text -> [passage_ids]}
        self.edge_metadata = {}  # {(source, target) -> [relationships]}

    def add_entity(
        self,
        canonical_text: str,
        entity_type: str,
        passage_ids: list[str],
    ):
        """
        Add an entity to the graph.
        
        Args:
            canonical_text: Canonical entity text
            entity_type: Type of entity (CONCEPT, PERSON, PRACTICE, STATE)
            passage_ids: List of passage IDs where entity appears
        """
        self.graph.add_node(
            canonical_text,
            type=entity_type,
            passage_count=len(passage_ids),
        )
        self.entity_to_passages[canonical_text] = passage_ids

    def add_relationship(
        self,
        source: str,
        relationship_type: str,
        target: str,
        weight: float = 1.0,
    ):
        """
        Add a relationship (edge) to the graph.
        
        Args:
            source: Source entity text
            relationship_type: Type of relationship (relates_to, leads_to, etc.)
            target: Target entity text
            weight: Edge weight (default 1.0)
        """
        # Add edge if both entities exist in graph
        if source in self.graph and target in self.graph:
            self.graph.add_edge(
                source,
                target,
                relationship=relationship_type,
                weight=weight,
            )

            # Track relationship metadata
            key = (source, target)
            if key not in self.edge_metadata:
                self.edge_metadata[key] = []
            self.edge_metadata[key].append(relationship_type)

    def build_from_canonical(
        self,
        canonical_map: dict[str, CanonicalEntity],
        relationships: list[Relationship],
    ):
        """
        Build graph from deduplicated entities and relationships.
        
        Args:
            canonical_map: Mapping from canonical text to CanonicalEntity
            relationships: List of remapped relationships
        """
        # Add all canonical entities as nodes
        for canonical_text, entity in canonical_map.items():
            self.add_entity(canonical_text, entity.entity_type, entity.passage_ids)

        # Add relationships as edges
        for rel in relationships:
            self.add_relationship(
                rel.source_entity,
                rel.relationship_type,
                rel.target_entity,
                weight=rel.confidence,
            )

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def traverse(
        self,
        start_entity: str,
        max_hops: int = 2,
        direction: str = "both",
    ) -> TraversalResult:
        """
        Traverse graph starting from an entity.
        
        Args:
            start_entity: Starting entity text
            max_hops: Maximum number of hops to traverse
            direction: "forward" (outgoing), "backward" (incoming), or "both"
            
        Returns:
            TraversalResult with traversed entities and connected passages
        """
        if start_entity not in self.graph:
            return TraversalResult(
                start_entity=start_entity,
                traversed_entities=[],
                edges_traversed=[],
                passages_reached=[],
            )

        traversed = {start_entity}
        edges = []
        queue = [(start_entity, 0)]  # (entity, hops_so_far)

        while queue:
            current, hops = queue.pop(0)

            if hops >= max_hops:
                continue

            # Forward traversal (outgoing edges)
            if direction in ["forward", "both"]:
                for neighbor in self.graph.successors(current):
                    if neighbor not in traversed:
                        traversed.add(neighbor)
                        edge_data = self.graph[current][neighbor]
                        edges.append((current, edge_data["relationship"], neighbor))
                        queue.append((neighbor, hops + 1))

            # Backward traversal (incoming edges)
            if direction in ["backward", "both"]:
                for predecessor in self.graph.predecessors(current):
                    if predecessor not in traversed:
                        traversed.add(predecessor)
                        edge_data = self.graph[predecessor][current]
                        edges.append((predecessor, edge_data["relationship"], current))
                        queue.append((predecessor, hops + 1))

        # Collect all passages connected to traversed entities
        passages = set()
        for entity in traversed:
            passages.update(self.entity_to_passages.get(entity, []))

        return TraversalResult(
            start_entity=start_entity,
            traversed_entities=list(traversed),
            edges_traversed=edges,
            passages_reached=list(passages),
        )

    def get_entity_info(self, entity_text: str) -> Optional[dict]:
        """
        Get information about an entity.
        
        Args:
            entity_text: Canonical entity text
            
        Returns:
            Dict with entity metadata or None if not found
        """
        if entity_text not in self.graph:
            return None

        node_data = self.graph.nodes[entity_text]
        neighbors_out = list(self.graph.successors(entity_text))
        neighbors_in = list(self.graph.predecessors(entity_text))

        return {
            "text": entity_text,
            "type": node_data.get("type"),
            "passage_count": node_data.get("passage_count", 0),
            "outgoing_neighbors": neighbors_out,
            "incoming_neighbors": neighbors_in,
            "passages": self.entity_to_passages.get(entity_text, []),
        }

    def search_entities(self, query: str) -> list[str]:
        """
        Search for entities by partial match.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching entity texts
        """
        query_lower = query.lower()
        matches = [
            entity
            for entity in self.graph.nodes()
            if query_lower in entity.lower()
        ]
        return matches

    def save(self, output_file: str):
        """
        Save graph to JSON.
        
        Args:
            output_file: Path to save JSON
        """
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        # Convert graph to JSON-serializable format
        nodes = [
            {
                "id": node,
                "type": self.graph.nodes[node].get("type"),
                "passage_count": self.graph.nodes[node].get("passage_count", 0),
            }
            for node in self.graph.nodes()
        ]

        edges = [
            {
                "source": source,
                "target": target,
                "relationship": data.get("relationship"),
                "weight": data.get("weight", 1.0),
            }
            for source, target, data in self.graph.edges(data=True)
        ]

        data = {
            "nodes": nodes,
            "edges": edges,
            "entity_passages": self.entity_to_passages,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Graph saved to {output_file}")

    @classmethod
    def load(cls, input_file: str) -> "KnowledgeGraph":
        """
        Load graph from JSON.
        
        Args:
            input_file: Path to JSON file
            
        Returns:
            KnowledgeGraph instance
        """
        with open(input_file) as f:
            data = json.load(f)

        kg = cls()

        # Add nodes
        for node_data in data.get("nodes", []):
            kg.graph.add_node(
                node_data["id"],
                type=node_data.get("type"),
                passage_count=node_data.get("passage_count", 0),
            )

        # Add edges
        for edge_data in data.get("edges", []):
            kg.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                relationship=edge_data.get("relationship"),
                weight=edge_data.get("weight", 1.0),
            )

        # Load entity-to-passages mapping
        kg.entity_to_passages = data.get("entity_passages", {})

        print(f"Graph loaded from {input_file}: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
        return kg

    def stats(self) -> dict:
        """
        Get graph statistics.
        
        Returns:
            Dict with graph metrics
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "num_connected_components": nx.number_weakly_connected_components(self.graph),
        }