"""
Local AI model engine for Mitra.

Uses a small language model (7B parameters) optimized for Indian context
and conversational AI without requiring external API calls.
"""

from typing import List, Dict, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from ..utils import get_logger

logger = get_logger(__name__)


class LocalModelEngine:
    """
    Local language model engine using Hugging Face transformers.

    Uses a 7B parameter model optimized for conversational AI with
    4-bit quantization for efficient inference on consumer hardware.

    Recommended models:
    - google/gemma-2-2b-it (lightweight, 2B params)
    - microsoft/Phi-3-mini-4k-instruct (3.8B params)
    - mistralai/Mistral-7B-Instruct-v0.3 (7B params, multilingual)
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Initialize the local model engine.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or 'auto')
            load_in_4bit: Use 4-bit quantization (reduces memory by ~75%)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info(
            "initializing_local_model",
            model=model_name,
            device=device,
            quantization="4bit" if load_in_4bit else "none",
        )

        # Configure quantization for memory efficiency
        quantization_config = None
        if load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with quantization
            model_kwargs = {"trust_remote_code": True}
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            # Move to device if not using auto device_map
            if not quantization_config and device != "auto":
                self.model = self.model.to(device)

            self.model.eval()  # Set to evaluation mode

            logger.info(
                "local_model_loaded_successfully",
                model=model_name,
                memory_gb=self._get_model_memory(),
            )

        except Exception as e:
            logger.error(
                "failed_to_load_local_model",
                model=model_name,
                error=str(e),
            )
            raise

    def _get_model_memory(self) -> float:
        """Get approximate model memory usage in GB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
            return 0.0
        except:
            return 0.0

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into prompt for the model.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        # Use chat template if available, otherwise simple format
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                pass

        # Fallback: simple formatting
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"

        formatted += "Assistant: "
        return formatted

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response from the model.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        try:
            # Format prompt
            prompt = self._format_messages(messages)

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            elif self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": True,
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode response (skip input tokens)
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Clean up response
            response = response.strip()

            logger.debug(
                "local_model_generated_response",
                input_length=input_length,
                output_length=len(response_tokens),
                response_preview=response[:100],
            )

            return response

        except Exception as e:
            logger.error(
                "local_model_generation_failed",
                error=str(e),
                model=self.model_name,
            )
            raise

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "memory_gb": self._get_model_memory(),
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
