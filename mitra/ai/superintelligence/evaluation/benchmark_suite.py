"""
🤖 Mitra AI - Benchmark Suite
Comprehensive AI evaluation benchmarks.
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


class BenchmarkType(Enum):
    """Types of benchmarks."""
    MMLU = "mmlu"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    ARC = "arc"
    HELLASWAG = "hellaswag"
    TRUTHFULQA = "truthfulqa"
    WINOGRANDE = "winogrande"


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    benchmark: str
    score: float
    accuracy: float
    total_samples: int
    correct: int
    incorrect: int
    skipped: int = 0
    time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkTargets:
    """Target scores for benchmarks."""
    mmlu: float = 0.90
    gsm8k: float = 0.95
    humaneval: float = 0.90
    arc: float = 0.90
    hellaswag: float = 0.90


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for AI evaluation.

    Features:
    - Multiple benchmark types
    - Parallel evaluation
    - Progress tracking
    - Result aggregation
    """

    def __init__(
        self,
        generate_fn: Optional[Callable] = None,
        targets: Optional[BenchmarkTargets] = None,
    ) -> None:
        self.generate_fn = generate_fn
        self.targets = targets or BenchmarkTargets()
        self._results: Dict[str, BenchmarkResult] = {}
        self._benchmarks: Dict[str, Any] = {}

    async def run_all(
        self,
        model: Any = None,
        tokenizer: Any = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run all benchmarks.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            num_samples: Optional limit on samples per benchmark

        Returns:
            Dict of benchmark results
        """
        logger.info("running_all_benchmarks")

        results = {}

        for benchmark_type in BenchmarkType:
            try:
                result = await self.run_benchmark(
                    benchmark_type,
                    model=model,
                    tokenizer=tokenizer,
                    num_samples=num_samples,
                )
                results[benchmark_type.value] = result
            except Exception as e:
                logger.error(
                    "benchmark_failed",
                    benchmark=benchmark_type.value,
                    error=str(e),
                )

        self._results = results
        return results

    async def run_benchmark(
        self,
        benchmark_type: BenchmarkType,
        model: Any = None,
        tokenizer: Any = None,
        num_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Run a specific benchmark.

        Args:
            benchmark_type: Type of benchmark to run
            model: The model to evaluate
            tokenizer: The tokenizer
            num_samples: Optional limit on samples

        Returns:
            BenchmarkResult
        """
        logger.info("running_benchmark", benchmark=benchmark_type.value)

        start_time = datetime.now(timezone.utc)

        # Import and run specific benchmark
        if benchmark_type == BenchmarkType.MMLU:
            from .benchmarks.mmlu import MMLUBenchmark
            benchmark = MMLUBenchmark()
        elif benchmark_type == BenchmarkType.GSM8K:
            from .benchmarks.gsm8k import GSM8KBenchmark
            benchmark = GSM8KBenchmark()
        elif benchmark_type == BenchmarkType.HUMANEVAL:
            from .benchmarks.humaneval import HumanEvalBenchmark
            benchmark = HumanEvalBenchmark()
        elif benchmark_type == BenchmarkType.ARC:
            from .benchmarks.arc import ARCBenchmark
            benchmark = ARCBenchmark()
        elif benchmark_type == BenchmarkType.HELLASWAG:
            from .benchmarks.hellaswag import HellaSwagBenchmark
            benchmark = HellaSwagBenchmark()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_type}")

        # Run evaluation
        result = await benchmark.evaluate(
            model=model,
            tokenizer=tokenizer,
            generate_fn=self.generate_fn,
            num_samples=num_samples,
        )

        # Calculate time
        result.time_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        logger.info(
            "benchmark_complete",
            benchmark=benchmark_type.value,
            score=result.score,
            time=result.time_seconds,
        )

        return result

    def check_targets(self) -> Dict[str, bool]:
        """Check if benchmark targets are met."""
        target_map = {
            "mmlu": self.targets.mmlu,
            "gsm8k": self.targets.gsm8k,
            "humaneval": self.targets.humaneval,
            "arc": self.targets.arc,
            "hellaswag": self.targets.hellaswag,
        }

        met = {}
        for name, target in target_map.items():
            result = self._results.get(name)
            if result:
                met[name] = result.score >= target
            else:
                met[name] = False

        return met

    def all_targets_met(self) -> bool:
        """Check if all targets are met."""
        return all(self.check_targets().values())

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        if not self._results:
            return {"status": "no_results"}

        scores = {name: r.score for name, r in self._results.items()}
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        targets_met = self.check_targets()

        return {
            "scores": scores,
            "average_score": avg_score,
            "targets_met": targets_met,
            "all_targets_met": all(targets_met.values()),
            "total_time_seconds": sum(r.time_seconds for r in self._results.values()),
        }

    def format_report(self) -> str:
        """Format a human-readable benchmark report."""
        lines = [
            "📊 *Benchmark Report*",
            "━━━━━━━━━━━━━━━━━━",
            "",
        ]

        target_map = {
            "mmlu": self.targets.mmlu,
            "gsm8k": self.targets.gsm8k,
            "humaneval": self.targets.humaneval,
            "arc": self.targets.arc,
            "hellaswag": self.targets.hellaswag,
        }

        for name, result in self._results.items():
            target = target_map.get(name, 0.9)
            met = result.score >= target
            emoji = "✅" if met else "❌"

            lines.append(f"{emoji} *{name.upper()}*")
            lines.append(f"   Score: `{result.score:.1%}` (target: {target:.0%})")
            lines.append(f"   Correct: {result.correct}/{result.total_samples}")
            lines.append(f"   Time: {result.time_seconds:.1f}s")
            lines.append("")

        summary = self.get_summary()
        lines.append(f"**Average Score:** `{summary['average_score']:.1%}`")
        lines.append(f"**All Targets Met:** {'✅ Yes' if summary['all_targets_met'] else '❌ No'}")
        lines.append("")
        lines.append("_Coded by Denvil with love 🤍_")

        return "\n".join(lines)

    async def quick_eval(
        self,
        model: Any = None,
        tokenizer: Any = None,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Quick evaluation with limited samples.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            num_samples: Samples per benchmark

        Returns:
            Dict of scores
        """
        results = await self.run_all(
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples,
        )

        return {name: r.score for name, r in results.items()}
