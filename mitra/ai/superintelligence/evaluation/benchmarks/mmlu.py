"""
🤖 Mitra AI - MMLU Benchmark
Massive Multitask Language Understanding evaluation.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..benchmark_suite import BenchmarkResult


class MMLUBenchmark:
    """
    MMLU (Massive Multitask Language Understanding) Benchmark.

    Tests knowledge across 57 subjects including:
    - STEM: Math, Physics, Chemistry, CS
    - Humanities: History, Philosophy, Law
    - Social Sciences: Psychology, Economics
    - Other: Professional knowledge, etc.
    """

    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]

    def __init__(self) -> None:
        self._dataset = None

    async def load_dataset(
        self,
        subjects: Optional[List[str]] = None,
        split: str = "test",
    ) -> None:
        """Load MMLU dataset."""
        try:
            from datasets import load_dataset

            self._dataset = load_dataset(
                "cais/mmlu",
                "all",
                split=split,
            )
            logger.info("mmlu_dataset_loaded", samples=len(self._dataset))

        except Exception as e:
            logger.error("mmlu_load_failed", error=str(e))
            self._dataset = None

    async def evaluate(
        self,
        model: Any = None,
        tokenizer: Any = None,
        generate_fn: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate model on MMLU.

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
                benchmark="mmlu",
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
            answer_idx = item["answer"]

            # Format as multiple choice
            prompt = self._format_question(question, choices)

            # Get model prediction
            if generate_fn:
                response = await generate_fn(prompt)
            else:
                response = "A"  # Default

            # Parse answer
            predicted = self._parse_answer(response)
            expected = ["A", "B", "C", "D"][answer_idx]

            if predicted == expected:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        return BenchmarkResult(
            benchmark="mmlu",
            score=accuracy,
            accuracy=accuracy,
            total_samples=total,
            correct=correct,
            incorrect=total - correct,
        )

    def _format_question(
        self,
        question: str,
        choices: List[str],
    ) -> str:
        """Format a question with choices."""
        formatted = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            letter = ["A", "B", "C", "D"][i]
            formatted += f"{letter}. {choice}\n"
        formatted += "\nAnswer with just the letter (A, B, C, or D):"
        return formatted

    def _parse_answer(self, response: str) -> str:
        """Parse the answer from model response."""
        response = response.strip().upper()

        # Look for letter at start
        for letter in ["A", "B", "C", "D"]:
            if response.startswith(letter):
                return letter

        # Look for letter anywhere
        for letter in ["A", "B", "C", "D"]:
            if letter in response:
                return letter

        return "A"  # Default
