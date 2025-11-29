#!/usr/bin/env python3
"""
🤖 Mitra AI - Benchmark Entry Point
CLI for running AI benchmarks.
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
@click.version_option(version="1.0.0", prog_name="Mitra AI Benchmark")
def cli():
    """🤖 Mitra AI Benchmark CLI - Coded by Denvil with love 🤍"""
    pass


@cli.command()
@click.option(
    "--model", "-m",
    default="microsoft/Phi-3-mini-4k-instruct",
    help="Model to evaluate"
)
@click.option(
    "--benchmark", "-b",
    multiple=True,
    default=["mmlu", "gsm8k", "humaneval", "arc", "hellaswag"],
    help="Benchmarks to run (can specify multiple)"
)
@click.option(
    "--samples", "-n",
    default=None,
    type=int,
    help="Number of samples per benchmark (for quick testing)"
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output file for results (JSON)"
)
@click.option(
    "--quantize/--no-quantize",
    default=True,
    help="Use 4-bit quantization"
)
def run(
    model: str,
    benchmark: tuple,
    samples: Optional[int],
    output: Optional[str],
    quantize: bool,
):
    """Run benchmarks on a model."""
    click.echo("🤖 Mitra AI Benchmark Suite")
    click.echo("=" * 50)
    click.echo(f"Model: {model}")
    click.echo(f"Benchmarks: {', '.join(benchmark)}")
    click.echo(f"Samples: {samples or 'All'}")
    click.echo(f"Quantized: {quantize}")
    click.echo("=" * 50)

    asyncio.run(_run_benchmarks(
        model=model,
        benchmarks=list(benchmark),
        samples=samples,
        output=output,
        quantize=quantize,
    ))


async def _run_benchmarks(
    model: str,
    benchmarks: list,
    samples: Optional[int],
    output: Optional[str],
    quantize: bool,
):
    """Run the benchmarks."""
    from mitra.ai.superintelligence.evaluation import BenchmarkSuite, BenchmarkResult
    from mitra.ai.superintelligence import MitraSuperBrain

    try:
        # Load model
        click.echo("\n📦 Loading model...")
        brain = MitraSuperBrain(
            model_name=model,
            load_in_4bit=quantize,
        )
        await brain.load_model()
        click.echo("✅ Model loaded")

        # Create benchmark suite
        async def generate_fn(prompt: str) -> str:
            result = await brain.think(prompt)
            return result.answer

        suite = BenchmarkSuite(generate_fn=generate_fn)

        # Run benchmarks
        click.echo("\n🏃 Running benchmarks...")
        results = await suite.run_all(num_samples=samples)

        # Display results
        click.echo("\n" + suite.format_report())

        # Save results if requested
        if output:
            import json
            with open(output, "w") as f:
                json.dump(
                    {name: r.__dict__ for name, r in results.items()},
                    f,
                    indent=2,
                    default=str,
                )
            click.echo(f"\n💾 Results saved to: {output}")

        # Check targets
        summary = suite.get_summary()
        if summary.get("all_targets_met"):
            click.echo("\n🎉 All benchmark targets met!")
        else:
            click.echo("\n⚠️ Some benchmark targets not met")
            for name, met in summary.get("targets_met", {}).items():
                if not met:
                    click.echo(f"   ❌ {name}")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}")
        logger.error("benchmark_error", error=str(e))
        sys.exit(1)


@cli.command()
@click.option(
    "--model", "-m",
    default="microsoft/Phi-3-mini-4k-instruct",
    help="Model to evaluate"
)
@click.option(
    "--samples", "-n",
    default=100,
    type=int,
    help="Number of samples per benchmark"
)
def quick(model: str, samples: int):
    """Run a quick benchmark with limited samples."""
    click.echo(f"⚡ Quick benchmark with {samples} samples per test")
    asyncio.run(_run_benchmarks(
        model=model,
        benchmarks=["mmlu", "gsm8k", "arc"],
        samples=samples,
        output=None,
        quantize=True,
    ))


@cli.command()
@click.option(
    "--benchmark", "-b",
    required=True,
    type=click.Choice(["mmlu", "gsm8k", "humaneval", "arc", "hellaswag"]),
    help="Specific benchmark to run"
)
@click.option(
    "--model", "-m",
    default="microsoft/Phi-3-mini-4k-instruct",
    help="Model to evaluate"
)
@click.option(
    "--samples", "-n",
    default=None,
    type=int,
    help="Number of samples"
)
def single(benchmark: str, model: str, samples: Optional[int]):
    """Run a single benchmark."""
    click.echo(f"📊 Running {benchmark.upper()} benchmark")
    asyncio.run(_run_benchmarks(
        model=model,
        benchmarks=[benchmark],
        samples=samples,
        output=None,
        quantize=True,
    ))


@cli.command()
def targets():
    """Show benchmark target scores."""
    click.echo("\n🎯 Benchmark Targets:")
    click.echo("=" * 40)
    click.echo("MMLU:      90%+ (Knowledge)")
    click.echo("GSM8K:     95%+ (Math reasoning)")
    click.echo("HumanEval: 90%+ (Code generation)")
    click.echo("ARC:       90%+ (Reasoning)")
    click.echo("HellaSwag: 90%+ (Commonsense)")
    click.echo("=" * 40)
    click.echo("\n_Coded by Denvil with love 🤍_")


@cli.command()
def list_benchmarks():
    """List available benchmarks."""
    click.echo("\n📋 Available Benchmarks:")
    click.echo("=" * 60)

    benchmarks = [
        ("MMLU", "Massive Multitask Language Understanding - 57 subjects"),
        ("GSM8K", "Grade School Math - 8K word problems"),
        ("HumanEval", "Code generation - 164 Python problems"),
        ("ARC", "AI2 Reasoning Challenge - Science questions"),
        ("HellaSwag", "Commonsense reasoning - Sentence completion"),
    ]

    for name, description in benchmarks:
        click.echo(f"\n📊 {name}")
        click.echo(f"   {description}")

    click.echo("\n_Coded by Denvil with love 🤍_")


if __name__ == "__main__":
    cli()
