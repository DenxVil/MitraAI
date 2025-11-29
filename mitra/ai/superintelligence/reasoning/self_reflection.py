"""
🤖 Mitra AI - Self-Reflection
Meta-cognitive reflection on reasoning quality.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ReflectionType(Enum):
    """Types of reflection."""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    CONFIDENCE = "confidence"


@dataclass
class ReflectionAspect:
    """A single aspect of reflection."""
    aspect_type: ReflectionType
    score: float  # 0-1
    feedback: str
    suggestion: Optional[str] = None


@dataclass
class ReflectionResult:
    """Result of self-reflection."""
    original_answer: str
    aspects: List[ReflectionAspect]
    overall_score: float
    needs_revision: bool
    revised_answer: Optional[str] = None
    revision_reasoning: Optional[str] = None


class SelfReflection:
    """
    Self-Reflection for meta-cognitive improvement.

    Features:
    - Multi-aspect evaluation
    - Weakness identification
    - Answer revision
    - Confidence calibration
    """

    def __init__(
        self,
        generate_fn: Optional[Callable] = None,
        revision_threshold: float = 0.6,
    ) -> None:
        self.generate_fn = generate_fn
        self.revision_threshold = revision_threshold

    async def reflect(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> ReflectionResult:
        """
        Reflect on an answer and potentially revise it.

        Args:
            question: The original question
            answer: The answer to reflect on
            context: Optional context

        Returns:
            ReflectionResult with evaluation and possible revision
        """
        # Evaluate each aspect
        aspects = []
        for aspect_type in ReflectionType:
            aspect = await self._evaluate_aspect(
                question, answer, aspect_type, context
            )
            aspects.append(aspect)

        # Calculate overall score
        overall_score = sum(a.score for a in aspects) / len(aspects)

        # Determine if revision is needed
        needs_revision = overall_score < self.revision_threshold

        # Revise if necessary
        revised_answer = None
        revision_reasoning = None
        if needs_revision:
            revised_answer, revision_reasoning = await self._revise_answer(
                question, answer, aspects
            )

        return ReflectionResult(
            original_answer=answer,
            aspects=aspects,
            overall_score=overall_score,
            needs_revision=needs_revision,
            revised_answer=revised_answer,
            revision_reasoning=revision_reasoning,
        )

    async def _evaluate_aspect(
        self,
        question: str,
        answer: str,
        aspect_type: ReflectionType,
        context: Optional[str],
    ) -> ReflectionAspect:
        """Evaluate a specific aspect of the answer."""
        prompts = {
            ReflectionType.CORRECTNESS: (
                f"Evaluate if this answer is factually correct.\n"
                f"Question: {question}\nAnswer: {answer}\n"
                f"Rate correctness (0-10) and explain:"
            ),
            ReflectionType.COMPLETENESS: (
                f"Evaluate if this answer is complete and thorough.\n"
                f"Question: {question}\nAnswer: {answer}\n"
                f"Rate completeness (0-10) and explain:"
            ),
            ReflectionType.CLARITY: (
                f"Evaluate if this answer is clear and well-explained.\n"
                f"Question: {question}\nAnswer: {answer}\n"
                f"Rate clarity (0-10) and explain:"
            ),
            ReflectionType.RELEVANCE: (
                f"Evaluate if this answer is relevant to the question.\n"
                f"Question: {question}\nAnswer: {answer}\n"
                f"Rate relevance (0-10) and explain:"
            ),
            ReflectionType.CONFIDENCE: (
                f"Evaluate the level of certainty in this answer.\n"
                f"Question: {question}\nAnswer: {answer}\n"
                f"Rate confidence appropriateness (0-10) and explain:"
            ),
        }

        prompt = prompts.get(aspect_type, prompts[ReflectionType.CORRECTNESS])

        if self.generate_fn:
            evaluation = await self.generate_fn(prompt)
            score, feedback, suggestion = self._parse_evaluation(evaluation)
        else:
            score = 0.7
            feedback = f"{aspect_type.value.capitalize()} evaluation pending."
            suggestion = None

        return ReflectionAspect(
            aspect_type=aspect_type,
            score=score,
            feedback=feedback,
            suggestion=suggestion,
        )

    def _parse_evaluation(
        self,
        evaluation: str,
    ) -> tuple:
        """Parse evaluation text to extract score and feedback."""
        # Try to extract a numeric score
        score = 0.7  # Default
        import re
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*(?:/\s*10|out of 10)?", evaluation)
        if numbers:
            try:
                extracted = float(numbers[0])
                if extracted <= 10:
                    score = extracted / 10
            except ValueError:
                pass

        # Feedback is the full evaluation
        feedback = evaluation[:200]

        # Look for suggestions
        suggestion = None
        if "improve" in evaluation.lower() or "suggest" in evaluation.lower():
            # Extract suggestion
            suggestion_idx = max(
                evaluation.lower().find("improve"),
                evaluation.lower().find("suggest"),
            )
            if suggestion_idx > 0:
                suggestion = evaluation[suggestion_idx:suggestion_idx + 150]

        return score, feedback, suggestion

    async def _revise_answer(
        self,
        question: str,
        answer: str,
        aspects: List[ReflectionAspect],
    ) -> tuple:
        """Revise the answer based on reflection feedback."""
        # Compile feedback
        feedback_summary = "\n".join(
            f"- {a.aspect_type.value}: {a.feedback}"
            for a in aspects if a.score < 0.7
        )

        suggestions = "\n".join(
            f"- {a.suggestion}"
            for a in aspects if a.suggestion
        )

        prompt = (
            f"Original question: {question}\n"
            f"Original answer: {answer}\n\n"
            f"Issues identified:\n{feedback_summary}\n\n"
            f"Suggestions:\n{suggestions}\n\n"
            f"Please provide an improved answer:"
        )

        if self.generate_fn:
            revised = await self.generate_fn(prompt)
            reasoning = f"Revised based on: {feedback_summary[:100]}"
        else:
            revised = answer
            reasoning = "Revision pending - generator not available"

        return revised, reasoning

    async def iterative_refinement(
        self,
        question: str,
        initial_answer: str,
        max_iterations: int = 3,
    ) -> ReflectionResult:
        """
        Iteratively refine an answer through multiple reflection cycles.

        Args:
            question: The original question
            initial_answer: The initial answer
            max_iterations: Maximum refinement iterations

        Returns:
            Final ReflectionResult
        """
        current_answer = initial_answer
        final_result = None

        for i in range(max_iterations):
            result = await self.reflect(question, current_answer)
            final_result = result

            if not result.needs_revision:
                logger.info(
                    "refinement_complete",
                    iterations=i + 1,
                    final_score=result.overall_score,
                )
                break

            if result.revised_answer:
                current_answer = result.revised_answer

        return final_result or ReflectionResult(
            original_answer=initial_answer,
            aspects=[],
            overall_score=0.5,
            needs_revision=False,
        )

    def format_result(self, result: ReflectionResult) -> str:
        """Format reflection result for display."""
        lines = ["🪞 *Self-Reflection Analysis*\n"]

        lines.append("**Aspect Scores:**")
        for aspect in result.aspects:
            emoji = "✅" if aspect.score >= 0.7 else "⚠️" if aspect.score >= 0.4 else "❌"
            lines.append(
                f"{emoji} {aspect.aspect_type.value.capitalize()}: "
                f"{aspect.score:.1%}"
            )

        lines.append(f"\n**Overall Score:** {result.overall_score:.1%}")
        lines.append(f"**Needs Revision:** {'Yes' if result.needs_revision else 'No'}")

        if result.revised_answer:
            lines.append(f"\n**Revised Answer:**\n{result.revised_answer[:500]}...")

        return "\n".join(lines)
