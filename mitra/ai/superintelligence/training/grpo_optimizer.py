"""
🤖 Mitra AI - GRPO Optimizer
Group Relative Policy Optimization for alignment.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    output_dir: str = "./outputs/grpo"

    # GRPO specific
    group_size: int = 4  # Number of responses per prompt
    kl_coef: float = 0.1  # KL divergence coefficient
    clip_range: float = 0.2  # PPO clip range
    value_clip_range: float = 0.2

    # Training
    num_epochs: int = 1
    batch_size: int = 4
    mini_batch_size: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 100

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Optimization
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0


class GRPOptimizer:
    """
    Group Relative Policy Optimization.

    GRPO improves on PPO by:
    - Computing rewards relative to group
    - Better sample efficiency
    - More stable training
    """

    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
    ) -> None:
        self.config = config or GRPOConfig()
        self._policy_model = None
        self._ref_model = None
        self._tokenizer = None
        self._reward_model = None

    async def setup(
        self,
        reward_fn: Optional[callable] = None,
    ) -> None:
        """Setup GRPO training."""
        logger.info("setting_up_grpo", model=self.config.model_name)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load policy model
            self._policy_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            # Load reference model (frozen)
            self._ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self._ref_model.eval()
            for param in self._ref_model.parameters():
                param.requires_grad = False

            # Set reward function
            if reward_fn:
                self._reward_model = reward_fn
            else:
                self._reward_model = self._default_reward

            logger.info("grpo_setup_complete")

        except Exception as e:
            logger.error("grpo_setup_failed", error=str(e))
            raise

    def _default_reward(
        self,
        prompt: str,
        response: str,
    ) -> float:
        """Default reward function based on length and coherence."""
        # Simple reward based on response quality
        score = 0.0

        # Length reward (prefer moderate length)
        length = len(response)
        if 50 <= length <= 500:
            score += 0.3
        elif length > 500:
            score += 0.2

        # Coherence reward (check for complete sentences)
        if response.endswith((".", "!", "?")):
            score += 0.2

        # No repetition penalty
        words = response.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        score += unique_ratio * 0.3

        return score

    async def generate_group(
        self,
        prompt: str,
    ) -> List[str]:
        """Generate a group of responses for a prompt."""
        if self._policy_model is None:
            raise RuntimeError("GRPO not setup. Call setup() first.")

        import torch

        responses = []

        # Tokenize prompt
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._policy_model.device)

        # Generate multiple responses
        for i in range(self.config.group_size):
            with torch.no_grad():
                outputs = self._policy_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature + (i * 0.05),  # Vary temperature
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)

        return responses

    async def compute_grpo_loss(
        self,
        prompts: List[str],
        responses_groups: List[List[str]],
        rewards_groups: List[List[float]],
    ) -> Dict[str, Any]:
        """
        Compute GRPO loss.

        Args:
            prompts: List of prompts
            responses_groups: List of response groups (one group per prompt)
            rewards_groups: List of reward groups (one group per prompt)

        Returns:
            Loss and metrics
        """
        import torch
        import torch.nn.functional as F

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        num_samples = 0

        for prompt, responses, rewards in zip(prompts, responses_groups, rewards_groups):
            # Compute relative rewards within group
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
            std_reward = max(std_reward, 1e-8)

            advantages = [(r - mean_reward) / std_reward for r in rewards]

            for response, advantage in zip(responses, advantages):
                # Get log probabilities
                policy_logp = await self._get_log_prob(prompt, response, self._policy_model)
                ref_logp = await self._get_log_prob(prompt, response, self._ref_model)

                # Policy loss (clipped)
                ratio = torch.exp(policy_logp - ref_logp)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

                # KL penalty
                kl_div = ref_logp - policy_logp

                total_policy_loss += policy_loss.mean()
                total_kl_loss += kl_div.mean() * self.config.kl_coef
                num_samples += 1

        if num_samples > 0:
            total_policy_loss /= num_samples
            total_kl_loss /= num_samples

        total_loss = total_policy_loss + total_kl_loss

        return {
            "loss": total_loss,
            "policy_loss": total_policy_loss,
            "kl_loss": total_kl_loss,
        }

    async def _get_log_prob(
        self,
        prompt: str,
        response: str,
        model: Any,
    ) -> Any:
        """Get log probability of response given prompt."""
        import torch

        full_text = prompt + response
        inputs = self._tokenizer(
            full_text,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Only consider response tokens
        prompt_length = len(self._tokenizer(prompt)["input_ids"])
        response_log_probs = token_log_probs[:, prompt_length - 1:]

        return response_log_probs.sum()

    async def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """
        Perform a single GRPO training step.

        Args:
            prompts: Batch of prompts

        Returns:
            Training metrics
        """
        # Generate response groups
        responses_groups = []
        rewards_groups = []

        for prompt in prompts:
            responses = await self.generate_group(prompt)
            rewards = [self._reward_model(prompt, r) for r in responses]
            responses_groups.append(responses)
            rewards_groups.append(rewards)

        # Compute loss
        loss_info = await self.compute_grpo_loss(prompts, responses_groups, rewards_groups)

        # Backward pass
        if hasattr(loss_info["loss"], "backward"):
            loss_info["loss"].backward()

        return {
            "loss": float(loss_info["loss"]) if hasattr(loss_info["loss"], "item") else loss_info["loss"],
            "policy_loss": float(loss_info["policy_loss"]) if hasattr(loss_info["policy_loss"], "item") else loss_info["policy_loss"],
            "kl_loss": float(loss_info["kl_loss"]) if hasattr(loss_info["kl_loss"], "item") else loss_info["kl_loss"],
            "mean_reward": sum(sum(g) for g in rewards_groups) / sum(len(g) for g in rewards_groups),
        }

    async def train(
        self,
        prompts: List[str],
        num_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train with GRPO.

        Args:
            prompts: Training prompts
            num_epochs: Number of epochs

        Returns:
            Training results
        """
        import torch

        epochs = num_epochs or self.config.num_epochs
        optimizer = torch.optim.AdamW(
            self._policy_model.parameters(),
            lr=self.config.learning_rate,
        )

        metrics = []
        logger.info("grpo_training_started", num_prompts=len(prompts))

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            num_batches = 0

            for i in range(0, len(prompts), self.config.batch_size):
                batch = prompts[i:i + self.config.batch_size]

                optimizer.zero_grad()
                step_metrics = await self.train_step(batch)

                if hasattr(step_metrics["loss"], "backward"):
                    # Already backpropagated in train_step
                    pass

                torch.nn.utils.clip_grad_norm_(
                    self._policy_model.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()

                epoch_loss += step_metrics["loss"]
                epoch_reward += step_metrics["mean_reward"]
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            avg_reward = epoch_reward / num_batches if num_batches > 0 else 0

            metrics.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "mean_reward": avg_reward,
            })

            logger.info(
                "grpo_epoch_complete",
                epoch=epoch + 1,
                loss=avg_loss,
                mean_reward=avg_reward,
            )

        logger.info("grpo_training_completed")

        return {
            "status": "completed",
            "metrics": metrics,
        }
