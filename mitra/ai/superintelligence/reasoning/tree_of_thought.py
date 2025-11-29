"""
🤖 Mitra AI - Tree of Thought Reasoning
Multi-path exploration for complex problem solving.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
import asyncio

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ThoughtNode:
    """A node in the thought tree."""
    node_id: str = field(default_factory=lambda: str(uuid4())[:8])
    content: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    is_terminal: bool = False
    is_solution: bool = False


@dataclass
class TreeResult:
    """Result of tree of thought exploration."""
    nodes: List[ThoughtNode]
    best_path: List[ThoughtNode]
    solution: str
    total_nodes_explored: int
    exploration_time_ms: float


class TreeOfThought:
    """
    Tree of Thought (ToT) reasoning implementation.

    Features:
    - Multi-path exploration
    - Breadth and depth-first search
    - Path evaluation and pruning
    - Best path selection
    """

    def __init__(
        self,
        generate_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        max_depth: int = 4,
        branching_factor: int = 3,
        beam_width: int = 2,
    ) -> None:
        self.generate_fn = generate_fn
        self.evaluate_fn = evaluate_fn
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width

    async def explore(
        self,
        problem: str,
        strategy: str = "bfs",
    ) -> TreeResult:
        """
        Explore solution space using tree of thought.

        Args:
            problem: The problem to solve
            strategy: Search strategy ("bfs" or "dfs")

        Returns:
            TreeResult with explored paths and best solution
        """
        start_time = datetime.now(timezone.utc)

        # Create root node
        root = ThoughtNode(
            content=f"Problem: {problem}",
            depth=0,
        )

        nodes = [root]

        # Explore based on strategy
        if strategy == "bfs":
            await self._bfs_explore(root, nodes, problem)
        else:
            await self._dfs_explore(root, nodes, problem, self.max_depth)

        # Find best path
        best_path = self._find_best_path(nodes)

        # Get solution from best terminal node
        terminal_nodes = [n for n in nodes if n.is_terminal]
        if terminal_nodes:
            best_terminal = max(terminal_nodes, key=lambda n: n.score)
            solution = best_terminal.content
        else:
            solution = best_path[-1].content if best_path else "No solution found"

        exploration_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return TreeResult(
            nodes=nodes,
            best_path=best_path,
            solution=solution,
            total_nodes_explored=len(nodes),
            exploration_time_ms=exploration_time,
        )

    async def _bfs_explore(
        self,
        root: ThoughtNode,
        all_nodes: List[ThoughtNode],
        problem: str,
    ) -> None:
        """Breadth-first search exploration."""
        current_level = [root]

        for depth in range(1, self.max_depth + 1):
            next_level = []

            for node in current_level:
                # Generate children
                children = await self._generate_children(node, problem)

                for child in children:
                    child.depth = depth
                    child.parent_id = node.node_id
                    node.children_ids.append(child.node_id)

                    # Evaluate child
                    child.score = await self._evaluate_node(child, problem)

                    all_nodes.append(child)
                    next_level.append(child)

            # Beam search pruning - keep only top k nodes
            next_level.sort(key=lambda n: n.score, reverse=True)
            current_level = next_level[:self.beam_width]

            # Mark terminal nodes
            if depth == self.max_depth:
                for node in current_level:
                    node.is_terminal = True

    async def _dfs_explore(
        self,
        node: ThoughtNode,
        all_nodes: List[ThoughtNode],
        problem: str,
        remaining_depth: int,
    ) -> None:
        """Depth-first search exploration."""
        if remaining_depth <= 0:
            node.is_terminal = True
            return

        # Generate children
        children = await self._generate_children(node, problem)

        for child in children:
            child.depth = node.depth + 1
            child.parent_id = node.node_id
            node.children_ids.append(child.node_id)

            # Evaluate child
            child.score = await self._evaluate_node(child, problem)

            all_nodes.append(child)

            # Recurse
            await self._dfs_explore(child, all_nodes, problem, remaining_depth - 1)

    async def _generate_children(
        self,
        node: ThoughtNode,
        problem: str,
    ) -> List[ThoughtNode]:
        """Generate child nodes (different approaches)."""
        children = []

        for i in range(self.branching_factor):
            prompt = (
                f"Problem: {problem}\n"
                f"Current thought: {node.content}\n"
                f"Generate approach {i + 1} (different from others):"
            )

            if self.generate_fn:
                content = await self.generate_fn(prompt)
            else:
                content = f"Approach {i + 1}: Alternative reasoning path..."

            child = ThoughtNode(content=content)
            children.append(child)

        return children

    async def _evaluate_node(
        self,
        node: ThoughtNode,
        problem: str,
    ) -> float:
        """Evaluate a node's promise."""
        if self.evaluate_fn:
            return await self.evaluate_fn(node.content, problem)

        # Simple heuristic: longer, more detailed thoughts score higher
        base_score = min(len(node.content) / 500, 1.0)
        depth_penalty = node.depth * 0.1
        return max(0.1, base_score - depth_penalty)

    def _find_best_path(
        self,
        nodes: List[ThoughtNode],
    ) -> List[ThoughtNode]:
        """Find the best path from root to terminal."""
        # Find best terminal node
        terminal_nodes = [n for n in nodes if n.is_terminal]
        if not terminal_nodes:
            return [nodes[0]] if nodes else []

        best_terminal = max(terminal_nodes, key=lambda n: n.score)

        # Trace back to root
        path = []
        current = best_terminal

        # Build node ID to node mapping
        node_map = {n.node_id: n for n in nodes}

        while current:
            path.append(current)
            if current.parent_id:
                current = node_map.get(current.parent_id)
            else:
                break

        path.reverse()
        return path

    def format_tree(self, result: TreeResult) -> str:
        """Format tree exploration for display."""
        lines = ["🌳 *Tree of Thought Exploration*\n"]

        # Show best path
        lines.append("**Best Path:**")
        for i, node in enumerate(result.best_path):
            indent = "  " * i
            score = f"[{node.score:.2f}]" if node.score > 0 else ""
            lines.append(f"{indent}→ {node.content[:80]}... {score}")

        lines.append(f"\n**Solution:** {result.solution}")
        lines.append(f"\n_Nodes explored: {result.total_nodes_explored}_")
        lines.append(f"_Time: {result.exploration_time_ms:.0f}ms_")

        return "\n".join(lines)
