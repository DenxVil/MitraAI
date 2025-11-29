"""
🤖 Mitra AI - HumanEval Benchmark
Code generation evaluation.
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


class HumanEvalBenchmark:
    """
    HumanEval Benchmark.

    Tests code generation ability with Python
    function completion tasks.
    """

    def __init__(self) -> None:
        self._dataset = None

    async def load_dataset(self) -> None:
        """Load HumanEval dataset."""
        try:
            from datasets import load_dataset

            self._dataset = load_dataset(
                "openai_humaneval",
                split="test",
            )
            logger.info("humaneval_dataset_loaded", samples=len(self._dataset))

        except Exception as e:
            logger.error("humaneval_load_failed", error=str(e))
            self._dataset = None

    async def evaluate(
        self,
        model: Any = None,
        tokenizer: Any = None,
        generate_fn: Optional[Callable] = None,
        num_samples: Optional[int] = None,
        k: int = 1,  # pass@k
    ) -> BenchmarkResult:
        """
        Evaluate model on HumanEval.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            generate_fn: Function to generate responses
            num_samples: Number of samples to evaluate
            k: Number of samples for pass@k

        Returns:
            BenchmarkResult
        """
        if self._dataset is None:
            await self.load_dataset()

        if self._dataset is None:
            return BenchmarkResult(
                benchmark="humaneval",
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
            prompt = item["prompt"]
            test_code = item["test"]
            entry_point = item["entry_point"]

            # Generate completion
            if generate_fn:
                response = await generate_fn(prompt)
            else:
                response = "    pass"

            # Extract code
            code = self._extract_code(prompt, response)

            # Test the code
            passed = await self._test_code(code, test_code, entry_point)

            if passed:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        return BenchmarkResult(
            benchmark="humaneval",
            score=accuracy,
            accuracy=accuracy,
            total_samples=total,
            correct=correct,
            incorrect=total - correct,
            details={"k": k},
        )

    def _extract_code(self, prompt: str, response: str) -> str:
        """Extract code from response."""
        # If response contains code blocks, extract them
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            return prompt + code_match.group(1)

        code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if code_match:
            return prompt + code_match.group(1)

        # Otherwise, assume response is the completion
        return prompt + response

    async def _test_code(
        self,
        code: str,
        test_code: str,
        entry_point: str,
    ) -> bool:
        """Test generated code."""
        # Combine code and tests
        full_code = code + "\n\n" + test_code

        try:
            # Execute in isolated namespace
            namespace: Dict[str, Any] = {}
            exec(full_code, namespace)

            # Run the check function
            check_fn = namespace.get("check")
            if check_fn:
                check_fn(namespace.get(entry_point))

            return True

        except Exception as e:
            logger.debug(
                "code_test_failed",
                error=str(e),
                entry_point=entry_point,
            )
            return False
