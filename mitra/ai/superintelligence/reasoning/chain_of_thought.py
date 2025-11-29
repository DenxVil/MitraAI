"""
🤖 Mitra AI - Chain of Thought Reasoning
Step-by-step reasoning for complex problems.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ThoughtStep:
    """A single step in the chain of thought."""
    step_number: int
    description: str
    content: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChainResult:
    """Result of chain of thought reasoning."""
    steps: List[ThoughtStep]
    conclusion: str
    total_confidence: float
    reasoning_time_ms: float


class ChainOfThought:
    """
    Chain of Thought (CoT) reasoning implementation.

    Features:
    - Step-by-step problem decomposition
    - Intermediate reasoning tracking
    - Confidence scoring
    - Prompting strategies
    """

    def __init__(
        self,
        generate_fn: Optional[Callable] = None,
        max_steps: int = 10,
        min_confidence: float = 0.3,
    ) -> None:
        self.generate_fn = generate_fn
        self.max_steps = max_steps
        self.min_confidence = min_confidence

    async def reason(
        self,
        problem: str,
        context: Optional[str] = None,
    ) -> ChainResult:
        """
        Apply chain of thought reasoning to a problem.

        Args:
            problem: The problem to solve
            context: Optional additional context

        Returns:
            ChainResult with steps and conclusion
        """
        start_time = datetime.now(timezone.utc)
        steps: List[ThoughtStep] = []

        # Step 1: Understand the problem
        step1 = await self._understand_problem(problem, context)
        steps.append(step1)

        # Step 2: Break down into sub-problems
        step2 = await self._decompose_problem(problem, step1.content)
        steps.append(step2)

        # Step 3-N: Solve each sub-problem
        sub_problems = self._extract_sub_problems(step2.content)
        for i, sub_problem in enumerate(sub_problems[:self.max_steps - 3]):
            solution = await self._solve_sub_problem(sub_problem, i + 3)
            steps.append(solution)

        # Final step: Synthesize conclusion
        conclusion_step = await self._synthesize_conclusion(problem, steps)
        steps.append(conclusion_step)

        # Calculate total confidence
        total_confidence = self._calculate_confidence(steps)

        # Calculate time
        reasoning_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return ChainResult(
            steps=steps,
            conclusion=conclusion_step.content,
            total_confidence=total_confidence,
            reasoning_time_ms=reasoning_time,
        )

    async def _understand_problem(
        self,
        problem: str,
        context: Optional[str],
    ) -> ThoughtStep:
        """Understand and restate the problem."""
        prompt = f"First, let me understand this problem:\n\n{problem}"
        if context:
            prompt += f"\n\nContext: {context}"

        if self.generate_fn:
            understanding = await self.generate_fn(prompt)
        else:
            understanding = f"Understanding: The problem asks about {problem[:50]}..."

        return ThoughtStep(
            step_number=1,
            description="Understanding the problem",
            content=understanding,
            confidence=0.9,
        )

    async def _decompose_problem(
        self,
        problem: str,
        understanding: str,
    ) -> ThoughtStep:
        """Break down the problem into smaller parts."""
        prompt = (
            f"Problem: {problem}\n\n"
            f"Understanding: {understanding}\n\n"
            f"Let me break this down into smaller steps:"
        )

        if self.generate_fn:
            decomposition = await self.generate_fn(prompt)
        else:
            decomposition = "Breaking down into sub-problems..."

        return ThoughtStep(
            step_number=2,
            description="Decomposing the problem",
            content=decomposition,
            confidence=0.85,
        )

    def _extract_sub_problems(self, decomposition: str) -> List[str]:
        """Extract sub-problems from decomposition."""
        # Simple extraction based on numbered lists or bullet points
        lines = decomposition.split("\n")
        sub_problems = []

        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                # Clean up the line
                clean = line.lstrip("0123456789.-•) ").strip()
                if clean:
                    sub_problems.append(clean)

        return sub_problems[:5]  # Limit to 5 sub-problems

    async def _solve_sub_problem(
        self,
        sub_problem: str,
        step_number: int,
    ) -> ThoughtStep:
        """Solve a single sub-problem."""
        prompt = f"Now solving: {sub_problem}"

        if self.generate_fn:
            solution = await self.generate_fn(prompt)
        else:
            solution = f"Solution for: {sub_problem}"

        return ThoughtStep(
            step_number=step_number,
            description=f"Solving: {sub_problem[:50]}...",
            content=solution,
            confidence=0.8,
        )

    async def _synthesize_conclusion(
        self,
        problem: str,
        steps: List[ThoughtStep],
    ) -> ThoughtStep:
        """Synthesize a conclusion from all steps."""
        steps_summary = "\n".join(
            f"Step {s.step_number}: {s.content[:100]}..."
            for s in steps
        )

        prompt = (
            f"Original problem: {problem}\n\n"
            f"Steps taken:\n{steps_summary}\n\n"
            f"Final conclusion:"
        )

        if self.generate_fn:
            conclusion = await self.generate_fn(prompt)
        else:
            conclusion = "Based on the reasoning steps, the answer is..."

        return ThoughtStep(
            step_number=len(steps) + 1,
            description="Final conclusion",
            content=conclusion,
            confidence=0.85,
        )

    def _calculate_confidence(self, steps: List[ThoughtStep]) -> float:
        """Calculate overall confidence from steps."""
        if not steps:
            return 0.0

        # Geometric mean of step confidences
        product = 1.0
        for step in steps:
            product *= step.confidence

        return product ** (1 / len(steps))

    def format_chain(self, result: ChainResult) -> str:
        """Format chain of thought for display."""
        lines = ["🧠 *Chain of Thought Reasoning*\n"]

        for step in result.steps:
            lines.append(f"**Step {step.step_number}:** {step.description}")
            lines.append(f"{step.content}\n")

        lines.append(f"\n**Conclusion:** {result.conclusion}")
        lines.append(f"\n_Confidence: {result.total_confidence:.1%}_")
        lines.append(f"_Time: {result.reasoning_time_ms:.0f}ms_")

        return "\n".join(lines)
