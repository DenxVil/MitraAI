#!/usr/bin/env python3
"""
🤖 Mitra AI - Improvement Loop Entry Point
CLI for iterative model improvement until targets are met.
Coded by Denvil with love 🤍
"""

import asyncio
import sys
from typing import Optional

try:
    import click
except ImportError:
    print("Click is required. Install with: pip install click")
    sys.exit(1)

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="Mitra AI Improvement")
def cli():
    """🤖 Mitra AI Improvement Loop - Coded by Denvil with love 🤍"""
    pass


@cli.command()
@click.option(
    "--model", "-m",
    default="microsoft/Phi-3-mini-4k-instruct",
    help="Base model to improve"
)
@click.option(
    "--dataset", "-d",
    default="openhermes",
    help="Training dataset"
)
@click.option(
    "--max-iterations", "-i",
    default=10,
    type=int,
    help="Maximum improvement iterations"
)
@click.option(
    "--samples-per-iter", "-s",
    default=1000,
    type=int,
    help="Training samples per iteration"
)
@click.option(
    "--eval-samples", "-e",
    default=500,
    type=int,
    help="Evaluation samples for benchmarks"
)
@click.option(
    "--output-dir", "-o",
    default="./outputs/improvement",
    help="Output directory"
)
@click.option(
    "--target-mmlu",
    default=0.90,
    type=float,
    help="Target MMLU score"
)
@click.option(
    "--target-gsm8k",
    default=0.95,
    type=float,
    help="Target GSM8K score"
)
@click.option(
    "--target-humaneval",
    default=0.90,
    type=float,
    help="Target HumanEval score"
)
@click.option(
    "--early-stop-patience",
    default=3,
    type=int,
    help="Iterations without improvement before stopping"
)
def run(
    model: str,
    dataset: str,
    max_iterations: int,
    samples_per_iter: int,
    eval_samples: int,
    output_dir: str,
    target_mmlu: float,
    target_gsm8k: float,
    target_humaneval: float,
    early_stop_patience: int,
):
    """Run the improvement loop until targets are met."""
    click.echo("🤖 Mitra AI Improvement Loop")
    click.echo("=" * 50)
    click.echo(f"Model: {model}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Max Iterations: {max_iterations}")
    click.echo(f"Samples/Iteration: {samples_per_iter}")
    click.echo(f"Eval Samples: {eval_samples}")
    click.echo("=" * 50)
    click.echo("Targets:")
    click.echo(f"  MMLU: {target_mmlu:.0%}")
    click.echo(f"  GSM8K: {target_gsm8k:.0%}")
    click.echo(f"  HumanEval: {target_humaneval:.0%}")
    click.echo("=" * 50)

    asyncio.run(_run_improvement(
        model=model,
        dataset=dataset,
        max_iterations=max_iterations,
        samples_per_iter=samples_per_iter,
        eval_samples=eval_samples,
        output_dir=output_dir,
        target_mmlu=target_mmlu,
        target_gsm8k=target_gsm8k,
        target_humaneval=target_humaneval,
        early_stop_patience=early_stop_patience,
    ))


async def _run_improvement(
    model: str,
    dataset: str,
    max_iterations: int,
    samples_per_iter: int,
    eval_samples: int,
    output_dir: str,
    target_mmlu: float,
    target_gsm8k: float,
    target_humaneval: float,
    early_stop_patience: int,
):
    """Run the improvement loop."""
    from mitra.ai.superintelligence.evaluation import (
        ImprovementLoop,
        ImprovementConfig,
        BenchmarkTargets,
    )
    from mitra.ai.superintelligence.training import QLoRATrainer, QLoRAConfig
    from mitra.ai.superintelligence.data import DataCollector, DataProcessor
    from mitra.ai.superintelligence import MitraSuperBrain

    try:
        # Load data
        click.echo("\n📦 Loading training data...")
        collector = DataCollector()
        train_data = await collector.download(dataset, split="train")

        processor = DataProcessor()
        processed_data = await processor.process_dataset(train_data)
        click.echo(f"✅ Loaded {len(processed_data)} training samples")

        # Load model
        click.echo("\n📦 Loading model...")
        brain = MitraSuperBrain(model_name=model, load_in_4bit=True)
        await brain.load_model()
        click.echo("✅ Model loaded")

        # Configure targets
        targets = BenchmarkTargets(
            mmlu=target_mmlu,
            gsm8k=target_gsm8k,
            humaneval=target_humaneval,
        )

        # Configure improvement
        config = ImprovementConfig(
            max_iterations=max_iterations,
            samples_per_iteration=samples_per_iter,
            eval_samples=eval_samples,
            early_stop_patience=early_stop_patience,
            targets=targets,
        )

        # Configure trainer
        trainer_config = QLoRAConfig(
            model_name=model,
            output_dir=output_dir,
        )
        trainer = QLoRATrainer(trainer_config)
        await trainer.prepare_model()

        # Run improvement loop
        click.echo("\n🚀 Starting improvement loop...")
        loop = ImprovementLoop(
            config=config,
            trainer=trainer,
        )

        result = await loop.run(
            model=brain._model,
            tokenizer=brain._tokenizer,
            train_dataset=processed_data,
        )

        # Display results
        click.echo("\n" + "=" * 50)
        click.echo("📊 Improvement Results")
        click.echo("=" * 50)

        if result["status"] == "completed":
            click.echo("🎉 All targets met!")
        else:
            click.echo("⚠️ Improvement completed but some targets not met")

        click.echo(f"\nIterations: {result['total_iterations']}")
        click.echo(f"Final Average Score: {result['final_avg_score']:.1%}")
        click.echo(f"Best Average Score: {result['best_avg_score']:.1%}")
        click.echo(f"Total Time: {result['total_time_seconds']:.0f}s")

        click.echo("\nFinal Scores:")
        for name, score in result.get("final_scores", {}).items():
            target = getattr(targets, name, 0.9)
            met = "✅" if score >= target else "❌"
            click.echo(f"  {met} {name.upper()}: {score:.1%} (target: {target:.0%})")

        click.echo("\n_Coded by Denvil with love 🤍_")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}")
        logger.error("improvement_error", error=str(e))
        sys.exit(1)


@cli.command()
def status():
    """Show current improvement status."""
    click.echo("📊 Improvement Status")
    click.echo("=" * 40)
    click.echo("No improvement loop currently running.")
    click.echo("\nUse 'improve run' to start the improvement loop.")


@cli.command()
def targets():
    """Show default target scores."""
    click.echo("\n🎯 Default Improvement Targets:")
    click.echo("=" * 40)
    click.echo("MMLU:      90%")
    click.echo("GSM8K:     95%")
    click.echo("HumanEval: 90%")
    click.echo("ARC:       90%")
    click.echo("HellaSwag: 90%")
    click.echo("=" * 40)
    click.echo("\nThese can be customized with --target-* options.")
    click.echo("\n_Coded by Denvil with love 🤍_")


if __name__ == "__main__":
    cli()
