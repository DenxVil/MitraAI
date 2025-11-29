"""
🤖 Mitra AI - Neural Router
Intelligent routing of queries to appropriate experts.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    target: str
    confidence: float
    reasoning: str
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


class NeuralRouter:
    """
    Neural router for intelligent query routing.

    Features:
    - Pattern-based routing
    - Semantic similarity (when model available)
    - Confidence scoring
    - Route history for learning
    """

    def __init__(
        self,
        use_neural: bool = False,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.use_neural = use_neural
        self.confidence_threshold = confidence_threshold
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._patterns: Dict[str, List[re.Pattern]] = {}
        self._keywords: Dict[str, List[str]] = {}
        self._history: List[Dict[str, Any]] = []

        # Default routes
        self._setup_default_routes()

    def _setup_default_routes(self) -> None:
        """Setup default routing patterns."""
        self.add_route(
            "mathematics",
            patterns=[
                r"\b(calculate|compute|solve|equation|math|formula)\b",
                r"\b(sum|product|divide|multiply|add|subtract)\b",
                r"\b\d+\s*[\+\-\*/\^]\s*\d+\b",
            ],
            keywords=["math", "calculate", "equation", "number", "algebra", "geometry"],
        )

        self.add_route(
            "coding",
            patterns=[
                r"\b(code|program|function|class|method|variable)\b",
                r"\b(python|javascript|java|c\+\+|rust|go)\b",
                r"\b(debug|error|bug|compile|run)\b",
            ],
            keywords=["code", "programming", "function", "debug", "algorithm"],
        )

        self.add_route(
            "reasoning",
            patterns=[
                r"\b(why|how|explain|reason|logic|because)\b",
                r"\b(if|then|therefore|hence|thus)\b",
                r"\b(analyze|compare|contrast|evaluate)\b",
            ],
            keywords=["reason", "explain", "logic", "analyze", "think"],
        )

        self.add_route(
            "creative",
            patterns=[
                r"\b(write|create|imagine|story|poem|song)\b",
                r"\b(creative|artistic|design|compose)\b",
            ],
            keywords=["write", "create", "story", "poem", "creative", "imagine"],
        )

        self.add_route(
            "science",
            patterns=[
                r"\b(science|physics|chemistry|biology|experiment)\b",
                r"\b(theory|hypothesis|research|study)\b",
            ],
            keywords=["science", "physics", "chemistry", "biology", "research"],
        )

        self.add_route(
            "general",
            patterns=[r".*"],
            keywords=["help", "what", "who", "when", "where"],
            priority=-1,  # Lowest priority as fallback
        )

    def add_route(
        self,
        name: str,
        patterns: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a routing rule."""
        self._routes[name] = {
            "priority": priority,
            "metadata": metadata or {},
        }

        if patterns:
            self._patterns[name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        if keywords:
            self._keywords[name] = [k.lower() for k in keywords]

        logger.debug("route_added", name=name)

    def remove_route(self, name: str) -> bool:
        """Remove a routing rule."""
        if name in self._routes:
            del self._routes[name]
            self._patterns.pop(name, None)
            self._keywords.pop(name, None)
            return True
        return False

    async def route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route a query to the appropriate target.

        Args:
            query: The query to route
            context: Optional context information

        Returns:
            RoutingDecision with target and confidence
        """
        scores = {}

        # Pattern matching
        for name, patterns in self._patterns.items():
            score = 0
            matches = 0
            for pattern in patterns:
                if pattern.search(query):
                    matches += 1
            if matches > 0:
                score = matches / len(patterns)
                scores[name] = scores.get(name, 0) + score * 0.5

        # Keyword matching
        query_lower = query.lower()
        for name, keywords in self._keywords.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                score = matches / len(keywords)
                scores[name] = scores.get(name, 0) + score * 0.3

        # Apply priority
        for name, route in self._routes.items():
            if name in scores:
                scores[name] += route["priority"] * 0.1

        # Neural routing if available
        if self.use_neural:
            neural_scores = await self._neural_route(query)
            for name, score in neural_scores.items():
                scores[name] = scores.get(name, 0) + score * 0.5

        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_scores:
            return RoutingDecision(
                target="general",
                confidence=0.0,
                reasoning="No matching route found",
            )

        best_target, best_score = sorted_scores[0]
        alternatives = sorted_scores[1:4]  # Top 3 alternatives

        decision = RoutingDecision(
            target=best_target,
            confidence=best_score,
            reasoning=self._generate_reasoning(query, best_target, best_score),
            alternatives=alternatives,
        )

        # Record history
        self._history.append({
            "query": query[:100],
            "decision": decision.target,
            "confidence": decision.confidence,
        })

        logger.debug(
            "routing_decision",
            target=decision.target,
            confidence=decision.confidence,
        )

        return decision

    async def _neural_route(
        self,
        query: str,
    ) -> Dict[str, float]:
        """Use neural network for routing (placeholder)."""
        # In production, use embedding similarity or classifier
        return {}

    def _generate_reasoning(
        self,
        query: str,
        target: str,
        confidence: float,
    ) -> str:
        """Generate reasoning for routing decision."""
        if confidence >= 0.8:
            return f"Strong match for {target} domain based on query patterns"
        elif confidence >= 0.5:
            return f"Moderate match for {target} domain"
        else:
            return f"Weak match for {target}, using as best available option"

    async def batch_route(
        self,
        queries: List[str],
    ) -> List[RoutingDecision]:
        """Route multiple queries efficiently."""
        return [await self.route(q) for q in queries]

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        if not self._history:
            return {"total_routes": 0, "distribution": {}}

        from collections import Counter
        targets = Counter(h["decision"] for h in self._history)

        return {
            "total_routes": len(self._history),
            "distribution": dict(targets),
            "avg_confidence": sum(h["confidence"] for h in self._history) / len(self._history),
        }

    def clear_history(self) -> None:
        """Clear routing history."""
        self._history.clear()
