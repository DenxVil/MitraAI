"""
🤖 Mitra AI - Mixture of Experts
Dynamic expert routing for specialized processing.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class Expert:
    """An expert model for specialized processing."""
    name: str
    domain: str
    model: Any = None
    weight: float = 1.0
    enabled: bool = True
    performance_score: float = 0.0

    async def process(
        self,
        input_data: Any,
        **kwargs: Any,
    ) -> Any:
        """Process input through this expert."""
        if not self.enabled or self.model is None:
            return None

        try:
            # Call the expert model
            if asyncio.iscoroutinefunction(self.model):
                result = await self.model(input_data, **kwargs)
            else:
                result = self.model(input_data, **kwargs)
            return result
        except Exception as e:
            logger.error("expert_process_error", expert=self.name, error=str(e))
            return None


class MixtureOfExperts:
    """
    Mixture of Experts (MoE) implementation.

    Features:
    - Dynamic expert selection
    - Load balancing
    - Performance tracking
    - Expert specialization
    """

    def __init__(
        self,
        top_k: int = 2,
        load_balance_factor: float = 0.1,
    ) -> None:
        self.top_k = top_k
        self.load_balance_factor = load_balance_factor
        self._experts: Dict[str, Expert] = {}
        self._router = None
        self._expert_usage: Dict[str, int] = {}

    def add_expert(self, expert: Expert) -> None:
        """Add an expert to the mixture."""
        self._experts[expert.name] = expert
        self._expert_usage[expert.name] = 0
        logger.info("expert_added", name=expert.name, domain=expert.domain)

    def remove_expert(self, name: str) -> bool:
        """Remove an expert from the mixture."""
        if name in self._experts:
            del self._experts[name]
            del self._expert_usage[name]
            logger.info("expert_removed", name=name)
            return True
        return False

    def get_expert(self, name: str) -> Optional[Expert]:
        """Get an expert by name."""
        return self._experts.get(name)

    def list_experts(self) -> List[Expert]:
        """List all experts."""
        return list(self._experts.values())

    async def route(
        self,
        input_data: Any,
        domain_hint: Optional[str] = None,
    ) -> List[Tuple[Expert, float]]:
        """
        Route input to the most appropriate experts.

        Args:
            input_data: The input to process
            domain_hint: Optional domain hint

        Returns:
            List of (expert, routing_weight) tuples
        """
        if not self._experts:
            return []

        # Get routing weights for each expert
        weights = await self._compute_routing_weights(input_data, domain_hint)

        # Sort by weight and select top-k
        sorted_experts = sorted(
            weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        selected = []
        for name, weight in sorted_experts[:self.top_k]:
            expert = self._experts.get(name)
            if expert and expert.enabled:
                selected.append((expert, weight))

        return selected

    async def _compute_routing_weights(
        self,
        input_data: Any,
        domain_hint: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute routing weights for each expert."""
        weights = {}

        for name, expert in self._experts.items():
            if not expert.enabled:
                continue

            # Base weight from expert configuration
            weight = expert.weight

            # Boost weight if domain matches hint
            if domain_hint and expert.domain == domain_hint:
                weight *= 2.0

            # Apply load balancing (reduce weight for overused experts)
            usage = self._expert_usage.get(name, 0)
            total_usage = sum(self._expert_usage.values()) or 1
            load_penalty = (usage / total_usage) * self.load_balance_factor
            weight *= (1 - load_penalty)

            # Apply performance score
            weight *= (0.5 + expert.performance_score * 0.5)

            weights[name] = weight

        # Normalize weights
        total = sum(weights.values()) or 1
        return {k: v / total for k, v in weights.items()}

    async def process(
        self,
        input_data: Any,
        domain_hint: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process input through selected experts.

        Args:
            input_data: The input to process
            domain_hint: Optional domain hint
            **kwargs: Additional arguments for experts

        Returns:
            Combined results from experts
        """
        # Route to experts
        selected = await self.route(input_data, domain_hint)

        if not selected:
            return {"error": "No experts available"}

        # Process through each selected expert
        results = {}
        for expert, weight in selected:
            result = await expert.process(input_data, **kwargs)
            if result is not None:
                results[expert.name] = {
                    "result": result,
                    "weight": weight,
                    "domain": expert.domain,
                }
                # Update usage
                self._expert_usage[expert.name] = self._expert_usage.get(expert.name, 0) + 1

        return results

    async def combine_results(
        self,
        results: Dict[str, Any],
        combination_strategy: str = "weighted",
    ) -> Any:
        """
        Combine results from multiple experts.

        Args:
            results: Results from experts
            combination_strategy: How to combine ("weighted", "best", "vote")

        Returns:
            Combined result
        """
        if not results:
            return None

        if combination_strategy == "best":
            # Return result with highest weight
            best = max(results.items(), key=lambda x: x[1]["weight"])
            return best[1]["result"]

        elif combination_strategy == "weighted":
            # Weighted combination (for numeric results)
            try:
                total_weight = sum(r["weight"] for r in results.values())
                combined = sum(
                    r["result"] * r["weight"] / total_weight
                    for r in results.values()
                )
                return combined
            except (TypeError, ValueError):
                # Fall back to best for non-numeric results
                return await self.combine_results(results, "best")

        elif combination_strategy == "vote":
            # Majority voting (for categorical results)
            from collections import Counter
            votes = Counter(r["result"] for r in results.values())
            return votes.most_common(1)[0][0]

        return list(results.values())[0]["result"]

    def update_performance(
        self,
        expert_name: str,
        score: float,
    ) -> None:
        """Update expert performance score."""
        expert = self._experts.get(expert_name)
        if expert:
            # Exponential moving average
            alpha = 0.1
            expert.performance_score = (
                alpha * score + (1 - alpha) * expert.performance_score
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get MoE statistics."""
        return {
            "total_experts": len(self._experts),
            "enabled_experts": len([e for e in self._experts.values() if e.enabled]),
            "usage": dict(self._expert_usage),
            "performance": {
                name: expert.performance_score
                for name, expert in self._experts.items()
            },
        }
