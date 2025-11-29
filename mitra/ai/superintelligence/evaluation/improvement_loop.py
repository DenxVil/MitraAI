"""
🤖 Mitra AI - Improvement Loop
Iterative training until benchmark targets are met.
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

from .benchmark_suite import BenchmarkSuite, BenchmarkTargets


@dataclass
class ImprovementConfig:
    """Configuration for improvement loop."""
    max_iterations: int = 10
    samples_per_iteration: int = 1000
    eval_samples: int = 500
    early_stop_patience: int = 3
    min_improvement: float = 0.01

    # Learning rate schedule
    initial_lr: float = 2e-4
    lr_decay: float = 0.9

    # Targets
    targets: BenchmarkTargets = field(default_factory=BenchmarkTargets)


@dataclass
class IterationResult:
    """Result from one improvement iteration."""
    iteration: int
    training_loss: float
    benchmark_scores: Dict[str, float]
    avg_score: float
    targets_met: Dict[str, bool]
    all_met: bool
    time_seconds: float
    learning_rate: float


class ImprovementLoop:
    """
    Iterative improvement loop for AI training.

    Features:
    - Train-evaluate-improve cycle
    - Automatic learning rate adjustment
    - Early stopping
    - Progress tracking
    """

    def __init__(
        self,
        config: Optional[ImprovementConfig] = None,
        trainer: Any = None,
        benchmark_suite: Optional[BenchmarkSuite] = None,
    ) -> None:
        self.config = config or ImprovementConfig()
        self.trainer = trainer
        self.benchmark_suite = benchmark_suite or BenchmarkSuite(
            targets=self.config.targets
        )
        self._iteration_results: List[IterationResult] = []
        self._best_score = 0.0
        self._no_improvement_count = 0
        self._current_lr = self.config.initial_lr

    async def run(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
    ) -> Dict[str, Any]:
        """
        Run the improvement loop.

        Args:
            model: Model to improve
            tokenizer: Tokenizer
            train_dataset: Training dataset

        Returns:
            Final results
        """
        logger.info(
            "improvement_loop_started",
            max_iterations=self.config.max_iterations,
        )

        start_time = datetime.now(timezone.utc)

        for iteration in range(1, self.config.max_iterations + 1):
            logger.info("iteration_started", iteration=iteration)

            # Run one iteration
            result = await self._run_iteration(
                iteration=iteration,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
            )

            self._iteration_results.append(result)

            # Check if all targets met
            if result.all_met:
                logger.info(
                    "all_targets_met",
                    iteration=iteration,
                    avg_score=result.avg_score,
                )
                break

            # Check for improvement
            if result.avg_score > self._best_score + self.config.min_improvement:
                self._best_score = result.avg_score
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1

            # Early stopping
            if self._no_improvement_count >= self.config.early_stop_patience:
                logger.info(
                    "early_stopping",
                    patience=self.config.early_stop_patience,
                    best_score=self._best_score,
                )
                break

            # Decay learning rate
            self._current_lr *= self.config.lr_decay

        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        return self._compile_results(total_time)

    async def _run_iteration(
        self,
        iteration: int,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
    ) -> IterationResult:
        """Run a single improvement iteration."""
        iter_start = datetime.now(timezone.utc)

        # Training phase
        training_loss = await self._train_iteration(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
        )

        # Evaluation phase
        benchmark_results = await self.benchmark_suite.run_all(
            model=model,
            tokenizer=tokenizer,
            num_samples=self.config.eval_samples,
        )

        scores = {name: r.score for name, r in benchmark_results.items()}
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        targets_met = self.benchmark_suite.check_targets()
        all_met = all(targets_met.values())

        iter_time = (datetime.now(timezone.utc) - iter_start).total_seconds()

        result = IterationResult(
            iteration=iteration,
            training_loss=training_loss,
            benchmark_scores=scores,
            avg_score=avg_score,
            targets_met=targets_met,
            all_met=all_met,
            time_seconds=iter_time,
            learning_rate=self._current_lr,
        )

        logger.info(
            "iteration_complete",
            iteration=iteration,
            avg_score=avg_score,
            all_met=all_met,
        )

        return result

    async def _train_iteration(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
    ) -> float:
        """Perform one training iteration."""
        if self.trainer is None:
            # Placeholder if no trainer provided
            return 0.0

        # Sample from dataset
        if hasattr(train_dataset, "shuffle"):
            train_subset = train_dataset.shuffle().select(
                range(min(self.config.samples_per_iteration, len(train_dataset)))
            )
        else:
            train_subset = train_dataset

        # Update trainer learning rate
        if hasattr(self.trainer, "config"):
            self.trainer.config.learning_rate = self._current_lr

        # Train
        result = await self.trainer.train(
            train_dataset=train_subset,
        )

        return result.get("training_loss", 0.0)

    def _compile_results(self, total_time: float) -> Dict[str, Any]:
        """Compile final results."""
        if not self._iteration_results:
            return {"status": "no_iterations", "total_time_seconds": total_time}

        final = self._iteration_results[-1]
        best_iteration = max(
            self._iteration_results,
            key=lambda r: r.avg_score,
        )

        return {
            "status": "completed" if final.all_met else "incomplete",
            "total_iterations": len(self._iteration_results),
            "final_scores": final.benchmark_scores,
            "final_avg_score": final.avg_score,
            "all_targets_met": final.all_met,
            "best_iteration": best_iteration.iteration,
            "best_avg_score": best_iteration.avg_score,
            "total_time_seconds": total_time,
            "iteration_history": [
                {
                    "iteration": r.iteration,
                    "avg_score": r.avg_score,
                    "training_loss": r.training_loss,
                    "learning_rate": r.learning_rate,
                }
                for r in self._iteration_results
            ],
        }

    def format_progress(self) -> str:
        """Format current progress for display."""
        if not self._iteration_results:
            return "No iterations completed yet."

        lines = [
            "📈 *Improvement Progress*",
            "━━━━━━━━━━━━━━━━━━",
            "",
        ]

        for result in self._iteration_results:
            emoji = "✅" if result.all_met else "🔄"
            lines.append(
                f"{emoji} Iteration {result.iteration}: "
                f"Score={result.avg_score:.1%}, "
                f"Loss={result.training_loss:.4f}"
            )

        latest = self._iteration_results[-1]
        lines.append("")
        lines.append("**Current Scores:**")
        for name, score in latest.benchmark_scores.items():
            met = latest.targets_met.get(name, False)
            emoji = "✅" if met else "❌"
            lines.append(f"  {emoji} {name}: {score:.1%}")

        lines.append("")
        lines.append(f"**Best Score:** {self._best_score:.1%}")
        lines.append(f"**Current LR:** {self._current_lr:.2e}")

        return "\n".join(lines)

    def get_iteration_results(self) -> List[IterationResult]:
        """Get all iteration results."""
        return self._iteration_results

    def reset(self) -> None:
        """Reset the improvement loop."""
        self._iteration_results.clear()
        self._best_score = 0.0
        self._no_improvement_count = 0
        self._current_lr = self.config.initial_lr
