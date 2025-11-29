"""
🤖 Mitra AI - Self-Consistency Reasoning
Voting across multiple reasoning paths.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import Counter

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ReasoningPath:
    """A single reasoning path."""
    path_id: int
    reasoning: str
    answer: str
    confidence: float


@dataclass
class ConsistencyResult:
    """Result of self-consistency reasoning."""
    paths: List[ReasoningPath]
    consensus_answer: str
    consensus_confidence: float
    vote_distribution: Dict[str, int]
    agreement_ratio: float


class SelfConsistency:
    """
    Self-Consistency reasoning implementation.

    Features:
    - Multiple reasoning paths
    - Answer extraction
    - Majority voting
    - Confidence calibration
    """

    def __init__(
        self,
        generate_fn: Optional[Callable] = None,
        num_paths: int = 5,
        temperature_range: Tuple[float, float] = (0.7, 1.0),
    ) -> None:
        self.generate_fn = generate_fn
        self.num_paths = num_paths
        self.temperature_range = temperature_range

    async def reason(
        self,
        problem: str,
        context: Optional[str] = None,
    ) -> ConsistencyResult:
        """
        Apply self-consistency reasoning.

        Args:
            problem: The problem to solve
            context: Optional context

        Returns:
            ConsistencyResult with consensus answer
        """
        paths: List[ReasoningPath] = []

        # Generate multiple reasoning paths
        for i in range(self.num_paths):
            temperature = self._get_temperature(i)
            path = await self._generate_path(
                problem, context, i, temperature
            )
            paths.append(path)

        # Extract and count answers
        answers = [p.answer for p in paths]
        vote_counts = Counter(answers)

        # Get consensus answer
        consensus_answer, consensus_count = vote_counts.most_common(1)[0]
        agreement_ratio = consensus_count / len(paths)

        # Calculate consensus confidence
        consensus_paths = [p for p in paths if p.answer == consensus_answer]
        avg_confidence = sum(p.confidence for p in consensus_paths) / len(consensus_paths)
        consensus_confidence = avg_confidence * agreement_ratio

        return ConsistencyResult(
            paths=paths,
            consensus_answer=consensus_answer,
            consensus_confidence=consensus_confidence,
            vote_distribution=dict(vote_counts),
            agreement_ratio=agreement_ratio,
        )

    def _get_temperature(self, path_index: int) -> float:
        """Get temperature for a given path index."""
        t_min, t_max = self.temperature_range
        if self.num_paths <= 1:
            return (t_min + t_max) / 2
        return t_min + (t_max - t_min) * path_index / (self.num_paths - 1)

    async def _generate_path(
        self,
        problem: str,
        context: Optional[str],
        path_id: int,
        temperature: float,
    ) -> ReasoningPath:
        """Generate a single reasoning path."""
        prompt = self._build_prompt(problem, context)

        if self.generate_fn:
            reasoning = await self.generate_fn(prompt, temperature=temperature)
        else:
            reasoning = f"Path {path_id}: Let me think through this step by step..."

        # Extract answer from reasoning
        answer = self._extract_answer(reasoning)

        # Estimate confidence
        confidence = self._estimate_confidence(reasoning)

        return ReasoningPath(
            path_id=path_id,
            reasoning=reasoning,
            answer=answer,
            confidence=confidence,
        )

    def _build_prompt(
        self,
        problem: str,
        context: Optional[str],
    ) -> str:
        """Build the prompt for reasoning."""
        prompt = (
            "Let's solve this problem step by step.\n\n"
            f"Problem: {problem}\n"
        )
        if context:
            prompt += f"\nContext: {context}\n"
        prompt += "\nThink through this carefully and give your final answer at the end."
        return prompt

    def _extract_answer(self, reasoning: str) -> str:
        """Extract the answer from reasoning text."""
        reasoning_lower = reasoning.lower()

        # Look for common answer indicators
        indicators = [
            "the answer is",
            "therefore,",
            "thus,",
            "so the answer is",
            "final answer:",
            "conclusion:",
            "in conclusion,",
        ]

        for indicator in indicators:
            if indicator in reasoning_lower:
                idx = reasoning_lower.find(indicator)
                answer_start = idx + len(indicator)
                answer = reasoning[answer_start:].strip()
                # Take until end of sentence
                for end_char in [".", "\n", "!"]:
                    if end_char in answer:
                        answer = answer[:answer.find(end_char)]
                return answer.strip()

        # If no indicator found, return last sentence
        sentences = reasoning.split(".")
        if sentences:
            return sentences[-1].strip() or sentences[-2].strip()

        return reasoning[-100:].strip()

    def _estimate_confidence(self, reasoning: str) -> float:
        """Estimate confidence based on reasoning quality."""
        # Simple heuristics
        score = 0.5

        # Longer reasoning suggests more thorough thinking
        if len(reasoning) > 200:
            score += 0.1
        if len(reasoning) > 500:
            score += 0.1

        # Presence of step indicators
        step_words = ["first", "second", "then", "next", "finally", "step"]
        step_count = sum(1 for w in step_words if w in reasoning.lower())
        score += min(step_count * 0.05, 0.2)

        # Presence of conclusion
        if any(w in reasoning.lower() for w in ["therefore", "thus", "conclusion"]):
            score += 0.1

        return min(score, 1.0)

    def format_result(self, result: ConsistencyResult) -> str:
        """Format result for display."""
        lines = ["🗳️ *Self-Consistency Voting*\n"]

        lines.append("**Vote Distribution:**")
        for answer, count in sorted(result.vote_distribution.items(), key=lambda x: -x[1]):
            pct = count / len(result.paths) * 100
            lines.append(f"• {answer}: {count} votes ({pct:.0f}%)")

        lines.append(f"\n**Consensus:** {result.consensus_answer}")
        lines.append(f"**Agreement:** {result.agreement_ratio:.1%}")
        lines.append(f"**Confidence:** {result.consensus_confidence:.1%}")

        return "\n".join(lines)
