"""
🤖 Mitra AI - Main Trainer
Training orchestrator with QLoRA support.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import asyncio

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model settings
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    output_dir: str = "./outputs"

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_steps: int = -1  # -1 for full training
    gradient_accumulation_steps: int = 4

    # Memory optimization
    use_qlora: bool = True
    use_gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10

    # Optimization
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class TrainingMetrics:
    """Metrics from training."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Trainer:
    """
    Main training orchestrator.

    Features:
    - QLoRA fine-tuning
    - Distributed training support
    - Checkpoint management
    - Metrics logging
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._metrics: List[TrainingMetrics] = []
        self._is_training = False

    async def setup(self) -> None:
        """Setup training environment."""
        logger.info(
            "setting_up_trainer",
            model=self.config.model_name,
            use_qlora=self.config.use_qlora,
        )

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            # Configure quantization for QLoRA
            if self.config.use_qlora:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Enable gradient checkpointing
            if self.config.use_gradient_checkpointing:
                self._model.gradient_checkpointing_enable()

            # Setup LoRA
            if self.config.use_qlora:
                await self._setup_lora()

            logger.info("trainer_setup_complete")

        except Exception as e:
            logger.error("trainer_setup_failed", error=str(e))
            raise

    async def _setup_lora(self) -> None:
        """Setup LoRA adapters."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # Prepare model for k-bit training
            self._model = prepare_model_for_kbit_training(self._model)

            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA
            self._model = get_peft_model(self._model, lora_config)

            # Print trainable parameters
            trainable_params = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self._model.parameters())

            logger.info(
                "lora_applied",
                trainable_params=trainable_params,
                total_params=total_params,
                percentage=trainable_params / total_params * 100,
            )

        except ImportError:
            logger.warning("peft_not_installed")

    async def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional training callbacks

        Returns:
            Training results
        """
        if self._model is None:
            await self.setup()

        self._is_training = True
        logger.info("training_started")

        try:
            from transformers import TrainingArguments, Trainer as HFTrainer

            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                optim=self.config.optim,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.max_grad_norm,
                evaluation_strategy="steps" if eval_dataset else "no",
                eval_steps=self.config.eval_steps if eval_dataset else None,
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                bf16=self.config.bf16,
                fp16=self.config.fp16,
                report_to="none",
                remove_unused_columns=False,
            )

            # Create trainer
            self._trainer = HFTrainer(
                model=self._model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self._tokenizer,
            )

            # Train
            train_result = self._trainer.train()

            # Save model
            self._trainer.save_model()

            logger.info(
                "training_completed",
                loss=train_result.training_loss,
            )

            return {
                "status": "completed",
                "training_loss": train_result.training_loss,
                "metrics": train_result.metrics,
            }

        except Exception as e:
            logger.error("training_failed", error=str(e))
            return {"status": "failed", "error": str(e)}

        finally:
            self._is_training = False

    async def save_checkpoint(
        self,
        path: Optional[str] = None,
    ) -> str:
        """Save a training checkpoint."""
        save_path = path or f"{self.config.output_dir}/checkpoint"

        if self._trainer:
            self._trainer.save_model(save_path)
        elif self._model:
            self._model.save_pretrained(save_path)
            if self._tokenizer:
                self._tokenizer.save_pretrained(save_path)

        logger.info("checkpoint_saved", path=save_path)
        return save_path

    async def load_checkpoint(
        self,
        path: str,
    ) -> None:
        """Load a training checkpoint."""
        logger.info("loading_checkpoint", path=path)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
            )

            logger.info("checkpoint_loaded")

        except Exception as e:
            logger.error("checkpoint_load_failed", error=str(e))
            raise

    def get_metrics(self) -> List[TrainingMetrics]:
        """Get training metrics."""
        return self._metrics

    @property
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training

    def stop_training(self) -> None:
        """Stop training gracefully."""
        if self._trainer:
            self._trainer.args.max_steps = 0
        self._is_training = False
        logger.info("training_stopped")
