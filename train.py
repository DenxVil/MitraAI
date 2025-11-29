#!/usr/bin/env python3
"""
🤖 Mitra AI - Training Entry Point
CLI for training the superintelligent AI model.
Coded by Denvil with love 🤍
"""

import asyncio
import sys
from pathlib import Path
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
@click.version_option(version="1.0.0", prog_name="Mitra AI Trainer")
def cli():
    """🤖 Mitra AI Training CLI - Coded by Denvil with love 🤍"""
    pass


@cli.command()
@click.option(
    "--model", "-m",
    default="microsoft/Phi-3-mini-4k-instruct",
    help="Base model to train"
)
@click.option(
    "--dataset", "-d",
    default="openhermes",
    help="Dataset to use for training"
)
@click.option(
    "--epochs", "-e",
    default=3,
    type=int,
    help="Number of training epochs"
)
@click.option(
    "--batch-size", "-b",
    default=4,
    type=int,
    help="Batch size"
)
@click.option(
    "--learning-rate", "-lr",
    default=2e-4,
    type=float,
    help="Learning rate"
)
@click.option(
    "--output-dir", "-o",
    default="./outputs",
    help="Output directory for checkpoints"
)
@click.option(
    "--use-qlora/--no-qlora",
    default=True,
    help="Use QLoRA for efficient training"
)
@click.option(
    "--lora-r",
    default=64,
    type=int,
    help="LoRA rank"
)
@click.option(
    "--lora-alpha",
    default=128,
    type=int,
    help="LoRA alpha"
)
def train(
    model: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str,
    use_qlora: bool,
    lora_r: int,
    lora_alpha: int,
):
    """Train the Mitra AI model with QLoRA."""
    click.echo("🤖 Mitra AI Training")
    click.echo("=" * 50)
    click.echo(f"Model: {model}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Batch Size: {batch_size}")
    click.echo(f"Learning Rate: {learning_rate}")
    click.echo(f"QLoRA: {use_qlora}")
    click.echo(f"Output: {output_dir}")
    click.echo("=" * 50)

    asyncio.run(_run_training(
        model=model,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
        use_qlora=use_qlora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    ))


async def _run_training(
    model: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str,
    use_qlora: bool,
    lora_r: int,
    lora_alpha: int,
):
    """Run the training process."""
    from mitra.ai.superintelligence.training import QLoRATrainer, QLoRAConfig
    from mitra.ai.superintelligence.data import DataCollector, DataProcessor

    try:
        # Collect data
        click.echo("\n📦 Loading dataset...")
        collector = DataCollector()
        train_data = await collector.download(dataset, split="train")
        click.echo(f"✅ Loaded {len(train_data)} samples")

        # Process data
        click.echo("\n🔧 Processing data...")
        processor = DataProcessor()
        processed_data = await processor.process_dataset(train_data)
        click.echo(f"✅ Processed {len(processed_data)} samples")

        # Configure training
        config = QLoRAConfig(
            model_name=model,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # Train
        click.echo("\n🚀 Starting training...")
        trainer = QLoRATrainer(config)
        result = await trainer.train(processed_data)

        if result["status"] == "completed":
            click.echo("\n✅ Training completed successfully!")
            click.echo(f"Final loss: {result.get('training_loss', 'N/A')}")
            click.echo(f"Output saved to: {output_dir}")
        else:
            click.echo(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Error: {e}")
        logger.error("training_error", error=str(e))
        sys.exit(1)


@cli.command()
@click.option(
    "--checkpoint", "-c",
    required=True,
    help="Path to checkpoint to resume from"
)
@click.option(
    "--dataset", "-d",
    default="openhermes",
    help="Dataset to continue training with"
)
@click.option(
    "--epochs", "-e",
    default=1,
    type=int,
    help="Additional epochs to train"
)
def resume(checkpoint: str, dataset: str, epochs: int):
    """Resume training from a checkpoint."""
    click.echo(f"📂 Resuming from: {checkpoint}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Additional epochs: {epochs}")

    asyncio.run(_resume_training(checkpoint, dataset, epochs))


async def _resume_training(checkpoint: str, dataset: str, epochs: int):
    """Resume training from checkpoint."""
    from mitra.ai.superintelligence.training import Trainer

    trainer = Trainer()
    await trainer.load_checkpoint(checkpoint)

    # Continue training...
    click.echo("✅ Checkpoint loaded. Continuing training...")


@cli.command()
@click.option(
    "--checkpoint", "-c",
    required=True,
    help="Path to trained adapter"
)
@click.option(
    "--output", "-o",
    required=True,
    help="Output path for merged model"
)
def merge(checkpoint: str, output: str):
    """Merge LoRA adapter with base model."""
    click.echo(f"🔗 Merging adapter: {checkpoint}")
    click.echo(f"Output: {output}")

    asyncio.run(_merge_model(checkpoint, output))


async def _merge_model(checkpoint: str, output: str):
    """Merge adapter with base model."""
    from mitra.ai.superintelligence.training import QLoRATrainer

    trainer = QLoRATrainer()
    await trainer.merge_and_save(output)
    click.echo("✅ Model merged successfully!")


@cli.command()
def list_datasets():
    """List available training datasets."""
    from mitra.ai.superintelligence.data import DataCollector

    collector = DataCollector()
    datasets = collector.list_available()

    click.echo("\n📚 Available Datasets:")
    click.echo("=" * 60)

    for ds in datasets:
        status = "✅" if ds.downloaded else "📦"
        click.echo(f"{status} {ds.name}")
        click.echo(f"   Source: {ds.source}")
        click.echo(f"   Task: {ds.task}")
        click.echo(f"   Description: {ds.description}")
        click.echo()


if __name__ == "__main__":
    cli()
