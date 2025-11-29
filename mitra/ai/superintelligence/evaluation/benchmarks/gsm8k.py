"""
🤖 Mitra AI - GSM8K Benchmark
Grade School Math evaluation.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable
import re

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..benchmark_suite import BenchmarkResult


class GSM8KBenchmark:
    """
    GSM8K (Grade School Math 8K) Benchmark.

    Tests mathematical reasoning with word problems
    requiring multi-step solutions.
    """

    def __init__(self) -> None:
        self._dataset = None

    async def load_dataset(
        self,
        split: str = "test",
    ) -> None:
        """Load GSM8K dataset."""
        try:
            from datasets import load_dataset

            self._dataset = load_dataset("gsm8k", "main", split=split)
            logger.info("gsm8k_dataset_loaded", samples=len(self._dataset))

        except Exception as e:
            logger.error("gsm8k_load_failed", error=str(e))
            self._dataset = None

    async def evaluate(
        self,
        model: Any = None,
        tokenizer: Any = None,
        generate_fn: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate model on GSM8K.

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
                benchmark="gsm8k",
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
            answer_text = item["answer"]

            # Extract ground truth answer
            expected = self._extract_answer(answer_text)

            # Format prompt
            prompt = self._format_prompt(question)

            # Get model response
            if generate_fn:
                response = await generate_fn(prompt)
            else:
                response = "0"

            # Extract predicted answer
            predicted = self._extract_answer(response)

            if predicted == expected:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        return BenchmarkResult(
            benchmark="gsm8k",
            score=accuracy,
            accuracy=accuracy,
            total_samples=total,
            correct=correct,
            incorrect=total - correct,
        )

    def _format_prompt(self, question: str) -> str:
        """Format a math problem prompt."""
        return (
            f"Solve this math problem step by step.\n\n"
            f"Problem: {question}\n\n"
            f"Think through each step carefully. "
            f"At the end, provide your final answer as a number after '#### '."
        )

    def _extract_answer(self, text: str) -> Optional[float]:
        """Extract numeric answer from text."""
        # Look for #### marker (GSM8K format)
        match = re.search(r"####\s*([\d,\.\-]+)", text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Look for "answer is" pattern
        match = re.search(r"answer\s+is\s+([\d,\.\-]+)", text.lower())
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Find all numbers and return the last one
        numbers = re.findall(r"[\d,]+\.?\d*", text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass

        return None
