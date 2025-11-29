"""
🤖 Mitra AI - HellaSwag Benchmark
Commonsense reasoning evaluation.
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


class HellaSwagBenchmark:
    """
    HellaSwag Benchmark.

    Tests commonsense reasoning with sentence
    completion tasks.
    """

    def __init__(self) -> None:
        self._dataset = None

    async def load_dataset(
        self,
        split: str = "validation",
    ) -> None:
        """Load HellaSwag dataset."""
        try:
            from datasets import load_dataset

            self._dataset = load_dataset(
                "Rowan/hellaswag",
                split=split,
            )
            logger.info("hellaswag_dataset_loaded", samples=len(self._dataset))

        except Exception as e:
            logger.error("hellaswag_load_failed", error=str(e))
            self._dataset = None

    async def evaluate(
        self,
        model: Any = None,
        tokenizer: Any = None,
        generate_fn: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate model on HellaSwag.

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
                benchmark="hellaswag",
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
            context = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])

            # Format question
            prompt = self._format_question(context, endings)

            # Get model response
            if generate_fn:
                response = await generate_fn(prompt)
            else:
                response = "A"

            # Parse answer
            predicted = self._parse_answer(response)

            if predicted == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        return BenchmarkResult(
            benchmark="hellaswag",
            score=accuracy,
            accuracy=accuracy,
            total_samples=total,
            correct=correct,
            incorrect=total - correct,
        )

    def _format_question(
        self,
        context: str,
        endings: List[str],
    ) -> str:
        """Format a completion question."""
        formatted = f"Complete this scenario:\n\n{context}\n\n"
        formatted += "Which ending is most appropriate?\n\n"

        letters = ["A", "B", "C", "D"]
        for i, ending in enumerate(endings):
            formatted += f"{letters[i]}. {ending}\n"

        formatted += "\nAnswer with just the letter (A, B, C, or D):"
        return formatted

    def _parse_answer(self, response: str) -> int:
        """Parse the answer from model response."""
        response = response.strip().upper()
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Look for letter at start
        for letter, idx in letter_to_idx.items():
            if response.startswith(letter):
                return idx

        # Look for letter anywhere
        for letter, idx in letter_to_idx.items():
            if letter in response:
                return idx

        return 0  # Default
