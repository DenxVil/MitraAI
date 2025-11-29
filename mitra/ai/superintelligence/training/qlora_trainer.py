"""
🤖 Mitra AI - QLoRA Trainer
Efficient 4-bit training with QLoRA.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training."""
    # Model
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    output_dir: str = "./outputs/qlora"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # nf4 or fp4
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Optimization
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3

    # Precision
    bf16: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100


class QLoRATrainer:
    """
    QLoRA (Quantized Low-Rank Adaptation) trainer.

    Features:
    - 4-bit quantization with NF4
    - Double quantization for memory efficiency
    - Paged optimizer for large models
    - Gradient checkpointing
    """

    def __init__(
        self,
        config: Optional[QLoRAConfig] = None,
    ) -> None:
        self.config = config or QLoRAConfig()
        self._model = None
        self._tokenizer = None
        self._peft_model = None
        self._training_start = None

    async def prepare_model(self) -> None:
        """Prepare model for QLoRA training."""
        logger.info(
            "preparing_qlora_model",
            model=self.config.model_name,
        )

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )

            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "right"

            # Load quantized model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # Required for quantized models with flash attention disabled
            )

            # Prepare for k-bit training
            self._model = prepare_model_for_kbit_training(
                self._model,
                use_gradient_checkpointing=True,
            )

            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA
            self._peft_model = get_peft_model(self._model, lora_config)

            # Log trainable parameters
            trainable, total = self._count_parameters()
            logger.info(
                "qlora_model_prepared",
                trainable_params=trainable,
                total_params=total,
                trainable_percentage=f"{trainable / total * 100:.2f}%",
            )

        except ImportError as e:
            logger.error("qlora_dependencies_missing", error=str(e))
            raise ImportError(
                "QLoRA requires: pip install peft bitsandbytes accelerate"
            )
        except Exception as e:
            logger.error("qlora_preparation_failed", error=str(e))
            raise

    def _count_parameters(self) -> tuple:
        """Count trainable and total parameters."""
        if self._peft_model is None:
            return 0, 0

        trainable = sum(
            p.numel() for p in self._peft_model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self._peft_model.parameters())

        return trainable, total

    async def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Train with QLoRA.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Training results
        """
        if self._peft_model is None:
            await self.prepare_model()

        self._training_start = datetime.now(timezone.utc)
        logger.info("qlora_training_started")

        try:
            from transformers import TrainingArguments, Trainer
            from trl import SFTTrainer

            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                optim=self.config.optim,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.max_grad_norm,
                bf16=self.config.bf16,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                evaluation_strategy="steps" if eval_dataset else "no",
                eval_steps=self.config.eval_steps if eval_dataset else None,
                save_total_limit=3,
                report_to="none",
            )

            # Create SFT trainer
            trainer = SFTTrainer(
                model=self._peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self._tokenizer,
                max_seq_length=self.config.max_seq_length,
            )

            # Train
            train_result = trainer.train()

            # Save
            trainer.save_model()
            self._tokenizer.save_pretrained(self.config.output_dir)

            training_time = (
                datetime.now(timezone.utc) - self._training_start
            ).total_seconds()

            logger.info(
                "qlora_training_completed",
                loss=train_result.training_loss,
                time_seconds=training_time,
            )

            return {
                "status": "completed",
                "training_loss": train_result.training_loss,
                "training_time_seconds": training_time,
                "metrics": train_result.metrics,
            }

        except ImportError:
            logger.error("trl_not_installed")
            return {"status": "failed", "error": "TRL library not installed"}
        except Exception as e:
            logger.error("qlora_training_failed", error=str(e))
            return {"status": "failed", "error": str(e)}

    async def save_adapter(
        self,
        path: Optional[str] = None,
    ) -> str:
        """Save LoRA adapter."""
        save_path = path or f"{self.config.output_dir}/adapter"

        if self._peft_model:
            self._peft_model.save_pretrained(save_path)
            if self._tokenizer:
                self._tokenizer.save_pretrained(save_path)

        logger.info("adapter_saved", path=save_path)
        return save_path

    async def merge_and_save(
        self,
        path: Optional[str] = None,
    ) -> str:
        """Merge LoRA adapter with base model and save."""
        save_path = path or f"{self.config.output_dir}/merged"

        if self._peft_model:
            merged_model = self._peft_model.merge_and_unload()
            merged_model.save_pretrained(save_path)
            if self._tokenizer:
                self._tokenizer.save_pretrained(save_path)

        logger.info("merged_model_saved", path=save_path)
        return save_path

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                return {
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                }
        except Exception:
            pass

        return {"allocated_gb": 0, "reserved_gb": 0}
