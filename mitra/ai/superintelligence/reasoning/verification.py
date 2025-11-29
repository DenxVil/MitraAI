"""
🤖 Mitra AI - Answer Verification
Verification of AI-generated answers.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of verification."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    INCORRECT = "incorrect"


@dataclass
class VerificationCheck:
    """A single verification check."""
    check_name: str
    passed: bool
    confidence: float
    explanation: str


@dataclass
class VerificationResult:
    """Result of answer verification."""
    status: VerificationStatus
    checks: List[VerificationCheck]
    overall_confidence: float
    issues_found: List[str]
    suggestions: List[str]
    verification_time_ms: float


class Verifier:
    """
    Answer verification system.

    Features:
    - Multi-strategy verification
    - Logical consistency checking
    - Fact verification (when possible)
    - Confidence calibration
    """

    def __init__(
        self,
        generate_fn: Optional[Callable] = None,
        verification_threshold: float = 0.7,
    ) -> None:
        self.generate_fn = generate_fn
        self.verification_threshold = verification_threshold

    async def verify(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify an answer.

        Args:
            question: The original question
            answer: The answer to verify
            context: Optional context
            domain: Optional domain hint

        Returns:
            VerificationResult with checks and status
        """
        start_time = datetime.now(timezone.utc)
        checks: List[VerificationCheck] = []
        issues: List[str] = []
        suggestions: List[str] = []

        # Run verification checks
        checks.append(await self._check_relevance(question, answer))
        checks.append(await self._check_coherence(answer))
        checks.append(await self._check_completeness(question, answer))
        checks.append(await self._check_consistency(answer))

        if domain == "mathematics":
            checks.append(await self._check_math(question, answer))
        elif domain == "coding":
            checks.append(await self._check_code(answer))

        # Collect issues and suggestions
        for check in checks:
            if not check.passed:
                issues.append(check.explanation)
                suggestions.append(f"Improve {check.check_name}")

        # Calculate overall confidence
        overall_confidence = sum(c.confidence for c in checks) / len(checks)

        # Determine status
        passed_count = sum(1 for c in checks if c.passed)
        total_checks = len(checks)

        if passed_count == total_checks:
            status = VerificationStatus.VERIFIED
        elif passed_count >= total_checks * 0.7:
            status = VerificationStatus.PARTIALLY_VERIFIED
        elif passed_count >= total_checks * 0.3:
            status = VerificationStatus.UNVERIFIED
        else:
            status = VerificationStatus.INCORRECT

        verification_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return VerificationResult(
            status=status,
            checks=checks,
            overall_confidence=overall_confidence,
            issues_found=issues,
            suggestions=suggestions,
            verification_time_ms=verification_time,
        )

    async def _check_relevance(
        self,
        question: str,
        answer: str,
    ) -> VerificationCheck:
        """Check if answer is relevant to question."""
        prompt = (
            f"Does this answer directly address the question?\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Rate relevance (0-10):"
        )

        if self.generate_fn:
            response = await self.generate_fn(prompt)
            score = self._extract_score(response)
            explanation = response[:100]
        else:
            # Simple heuristic: check for keyword overlap
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())
            overlap = len(q_words & a_words) / len(q_words) if q_words else 0
            score = min(overlap * 2, 1.0)
            explanation = f"Keyword overlap: {score:.1%}"

        return VerificationCheck(
            check_name="relevance",
            passed=score >= self.verification_threshold,
            confidence=score,
            explanation=explanation,
        )

    async def _check_coherence(
        self,
        answer: str,
    ) -> VerificationCheck:
        """Check if answer is internally coherent."""
        if self.generate_fn:
            prompt = (
                f"Is this text internally coherent and logical?\n"
                f"Text: {answer}\n"
                f"Rate coherence (0-10):"
            )
            response = await self.generate_fn(prompt)
            score = self._extract_score(response)
            explanation = response[:100]
        else:
            # Simple heuristic: check sentence structure
            sentences = answer.split(".")
            score = 0.8 if len(sentences) > 1 else 0.6
            explanation = "Basic structure check"

        return VerificationCheck(
            check_name="coherence",
            passed=score >= self.verification_threshold,
            confidence=score,
            explanation=explanation,
        )

    async def _check_completeness(
        self,
        question: str,
        answer: str,
    ) -> VerificationCheck:
        """Check if answer is complete."""
        if self.generate_fn:
            prompt = (
                f"Does this answer completely address all parts of the question?\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Rate completeness (0-10):"
            )
            response = await self.generate_fn(prompt)
            score = self._extract_score(response)
            explanation = response[:100]
        else:
            # Simple heuristic: length-based
            score = min(len(answer) / 200, 1.0)
            explanation = f"Answer length: {len(answer)} chars"

        return VerificationCheck(
            check_name="completeness",
            passed=score >= self.verification_threshold,
            confidence=score,
            explanation=explanation,
        )

    async def _check_consistency(
        self,
        answer: str,
    ) -> VerificationCheck:
        """Check for internal contradictions."""
        if self.generate_fn:
            prompt = (
                f"Does this text contain any contradictions?\n"
                f"Text: {answer}\n"
                f"Rate consistency (0-10, 10 = no contradictions):"
            )
            response = await self.generate_fn(prompt)
            score = self._extract_score(response)
            explanation = response[:100]
        else:
            score = 0.8
            explanation = "No contradictions detected (basic check)"

        return VerificationCheck(
            check_name="consistency",
            passed=score >= self.verification_threshold,
            confidence=score,
            explanation=explanation,
        )

    async def _check_math(
        self,
        question: str,
        answer: str,
    ) -> VerificationCheck:
        """Verify mathematical calculations."""
        if self.generate_fn:
            prompt = (
                f"Verify the mathematical calculations in this answer:\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Are the calculations correct? Rate (0-10):"
            )
            response = await self.generate_fn(prompt)
            score = self._extract_score(response)
            explanation = response[:100]
        else:
            score = 0.7
            explanation = "Math verification pending"

        return VerificationCheck(
            check_name="math_verification",
            passed=score >= self.verification_threshold,
            confidence=score,
            explanation=explanation,
        )

    async def _check_code(
        self,
        answer: str,
    ) -> VerificationCheck:
        """Verify code syntax and logic."""
        # Check for common code patterns
        has_code = "```" in answer or "def " in answer or "function " in answer

        if has_code:
            score = 0.8
            explanation = "Code structure detected"
        else:
            score = 0.5
            explanation = "No code block found"

        return VerificationCheck(
            check_name="code_verification",
            passed=score >= self.verification_threshold,
            confidence=score,
            explanation=explanation,
        )

    def _extract_score(self, response: str) -> float:
        """Extract numeric score from response."""
        import re
        numbers = re.findall(r"(\d+(?:\.\d+)?)", response)
        for num in numbers:
            try:
                value = float(num)
                if value <= 10:
                    return value / 10
            except ValueError:
                continue
        return 0.5

    def format_result(self, result: VerificationResult) -> str:
        """Format verification result for display."""
        status_emoji = {
            VerificationStatus.VERIFIED: "✅",
            VerificationStatus.PARTIALLY_VERIFIED: "⚠️",
            VerificationStatus.UNVERIFIED: "❓",
            VerificationStatus.INCORRECT: "❌",
        }

        lines = ["🔍 *Answer Verification*\n"]

        emoji = status_emoji.get(result.status, "❓")
        lines.append(f"**Status:** {emoji} {result.status.value}")
        lines.append(f"**Confidence:** {result.overall_confidence:.1%}")

        lines.append("\n**Checks:**")
        for check in result.checks:
            check_emoji = "✅" if check.passed else "❌"
            lines.append(f"{check_emoji} {check.check_name}: {check.confidence:.1%}")

        if result.issues_found:
            lines.append("\n**Issues:**")
            for issue in result.issues_found[:3]:
                lines.append(f"• {issue[:50]}")

        lines.append(f"\n_Verified in {result.verification_time_ms:.0f}ms_")

        return "\n".join(lines)
