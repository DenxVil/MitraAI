"""
🤖 Mitra AI - ARC Benchmark
AI2 Reasoning Challenge evaluation.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..benchmark_suite import BenchmarkResult


class ARCBenchmark:
    """
    ARC (AI2 Reasoning Challenge) Benchmark.

    Tests reasoning ability with science questions
    requiring multi-step reasoning.
    """

    def __init__(self, difficulty: str = "challenge") -> None:
        """
        Initialize ARC benchmark.

        Args:
            difficulty: "easy" or "challenge"
        """
        self.difficulty = difficulty
        self._dataset = None

    async def load_dataset(
        self,
        split: str = "test",
    ) -> None:
        """Load ARC dataset."""
        try:
            from datasets import load_dataset

            config = "ARC-Challenge" if self.difficulty == "challenge" else "ARC-Easy"

            self._dataset = load_dataset(
                "allenai/ai2_arc",
                config,
                split=split,
            )
            logger.info(
                "arc_dataset_loaded",
                samples=len(self._dataset),
                difficulty=self.difficulty,
            )

        except Exception as e:
            logger.error("arc_load_failed", error=str(e))
            self._dataset = None

    async def evaluate(
        self,
        model: Any = None,
        tokenizer: Any = None,
        generate_fn: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate model on ARC.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            generate_fn: Function to generate responses
            num_samples: Number of samples to evaluate

        Returns:
            BenchmarkResult
        """
        if self._dataset is None:
            await self.load_dataset()

        if self._dataset is None:
            return BenchmarkResult(
                benchmark="arc",
                score=0.0,
                accuracy=0.0,
                total_samples=0,
                correct=0,
                incorrect=0,
            )

        # Sample if needed
        samples = self._dataset
        if num_samples and num_samples < len(samples):
            samples = samples.shuffle().select(range(num_samples))

        correct = 0
        total = 0

        for item in samples:
            question = item["question"]
            choices = item["choices"]
            answer_key = item["answerKey"]

            # Format question
            prompt = self._format_question(question, choices)

            # Get model response
            if generate_fn:
                response = await generate_fn(prompt)
            else:
                response = "A"

            # Parse answer
            predicted = self._parse_answer(response, choices)

            if predicted == answer_key:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        return BenchmarkResult(
            benchmark="arc",
            score=accuracy,
            accuracy=accuracy,
            total_samples=total,
            correct=correct,
            incorrect=total - correct,
            details={"difficulty": self.difficulty},
        )

    def _format_question(
        self,
        question: str,
        choices: Dict[str, Any],
    ) -> str:
        """Format a question with choices."""
        formatted = f"Question: {question}\n\n"

        labels = choices["label"]
        texts = choices["text"]

        for label, text in zip(labels, texts):
            formatted += f"{label}. {text}\n"

        formatted += "\nAnswer with just the letter:"
        return formatted

    def _parse_answer(
        self,
        response: str,
        choices: Dict[str, Any],
    ) -> str:
        """Parse the answer from model response."""
        response = response.strip().upper()
        labels = choices["label"]

        # Look for label at start
        for label in labels:
            if response.startswith(label):
                return label

        # Look for label anywhere
        for label in labels:
            if label in response:
                return label

        return labels[0] if labels else "A"  # Default
