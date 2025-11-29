"""
🤖 Mitra AI - Data Processor
Clean and format data for training.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    # Text cleaning
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_whitespace: bool = True
    max_length: int = 4096
    min_length: int = 10

    # Formatting
    add_eos_token: bool = True
    chat_template: str = "chatml"  # chatml, alpaca, vicuna

    # Filtering
    remove_duplicates: bool = True
    remove_empty: bool = True
    language_filter: Optional[str] = None  # e.g., "en"

    # Quality
    min_quality_score: float = 0.5
    check_quality: bool = True


class DataProcessor:
    """
    Process and clean datasets for training.

    Features:
    - Text cleaning and normalization
    - Format conversion (ChatML, Alpaca, etc.)
    - Quality filtering
    - Deduplication
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
    ) -> None:
        self.config = config or ProcessingConfig()
        self._processed_count = 0
        self._filtered_count = 0

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove emails
        if self.config.remove_emails:
            text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

        # Truncate if too long
        if len(text) > self.config.max_length:
            text = text[: self.config.max_length]

        return text

    def format_conversation(
        self,
        messages: List[Dict[str, str]],
        template: Optional[str] = None,
    ) -> str:
        """
        Format a conversation using specified template.

        Args:
            messages: List of {"role": ..., "content": ...} messages
            template: Chat template to use

        Returns:
            Formatted conversation string
        """
        template = template or self.config.chat_template

        if template == "chatml":
            return self._format_chatml(messages)
        elif template == "alpaca":
            return self._format_alpaca(messages)
        elif template == "vicuna":
            return self._format_vicuna(messages)
        else:
            return self._format_simple(messages)

    def _format_chatml(self, messages: List[Dict[str, str]]) -> str:
        """Format as ChatML."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = self.clean_text(msg.get("content", ""))
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted

    def _format_alpaca(self, messages: List[Dict[str, str]]) -> str:
        """Format as Alpaca instruction format."""
        instruction = ""
        input_text = ""
        output = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = self.clean_text(msg.get("content", ""))

            if role == "system":
                instruction = content
            elif role == "user":
                if instruction:
                    input_text = content
                else:
                    instruction = content
            elif role == "assistant":
                output = content

        parts = []
        if instruction:
            parts.append(f"### Instruction:\n{instruction}")
        if input_text:
            parts.append(f"### Input:\n{input_text}")
        if output:
            parts.append(f"### Response:\n{output}")

        return "\n\n".join(parts)

    def _format_vicuna(self, messages: List[Dict[str, str]]) -> str:
        """Format as Vicuna style."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = self.clean_text(msg.get("content", ""))

            if role == "user":
                formatted += f"USER: {content}\n"
            elif role == "assistant":
                formatted += f"ASSISTANT: {content}\n"
            elif role == "system":
                formatted += f"SYSTEM: {content}\n"

        return formatted

    def _format_simple(self, messages: List[Dict[str, str]]) -> str:
        """Simple concatenation format."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = self.clean_text(msg.get("content", ""))
            parts.append(f"[{role.upper()}]: {content}")
        return "\n\n".join(parts)

    def estimate_quality(self, text: str) -> float:
        """
        Estimate text quality score.

        Returns:
            Quality score between 0 and 1
        """
        if not text:
            return 0.0

        score = 0.5  # Base score

        # Length check
        length = len(text)
        if self.config.min_length <= length <= self.config.max_length:
            score += 0.1
        elif length < self.config.min_length:
            score -= 0.3

        # Check for complete sentences
        if text.endswith((".", "!", "?")):
            score += 0.1

        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in "!?.,;:") / len(text)
        if punct_ratio > 0.1:
            score -= 0.1

        # Check for word diversity
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.2

        # Check for code (if present, should be well-formatted)
        if "```" in text:
            code_blocks = text.count("```")
            if code_blocks % 2 == 0:
                score += 0.1
            else:
                score -= 0.1

        return max(0.0, min(1.0, score))

    async def process_dataset(
        self,
        dataset: Any,
        text_column: str = "text",
        output_column: str = "formatted_text",
    ) -> Any:
        """
        Process an entire dataset.

        Args:
            dataset: HuggingFace dataset
            text_column: Name of text column
            output_column: Name for output column

        Returns:
            Processed dataset
        """
        logger.info("processing_dataset", samples=len(dataset))

        def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
            text = example.get(text_column, "")

            # Handle conversation format
            if isinstance(text, list):
                formatted = self.format_conversation(text)
            else:
                formatted = self.clean_text(text)

            # Add EOS token
            if self.config.add_eos_token and formatted:
                formatted += "<|endoftext|>"

            example[output_column] = formatted
            example["quality_score"] = self.estimate_quality(formatted)

            return example

        # Apply processing
        processed = dataset.map(process_example)

        # Filter by quality
        if self.config.check_quality:
            original_len = len(processed)
            processed = processed.filter(
                lambda x: x.get("quality_score", 0) >= self.config.min_quality_score
            )
            filtered_len = len(processed)
            self._filtered_count += original_len - filtered_len
            logger.info(
                "quality_filtered",
                removed=original_len - filtered_len,
                remaining=filtered_len,
            )

        # Remove empty
        if self.config.remove_empty:
            processed = processed.filter(
                lambda x: len(x.get(output_column, "")) >= self.config.min_length
            )

        self._processed_count += len(processed)

        logger.info("processing_complete", samples=len(processed))

        return processed

    async def process_conversations(
        self,
        dataset: Any,
        conversations_column: str = "conversations",
    ) -> Any:
        """
        Process a dataset with conversation format.

        Args:
            dataset: HuggingFace dataset
            conversations_column: Column containing conversations

        Returns:
            Processed dataset
        """

        def process_conv(example: Dict[str, Any]) -> Dict[str, Any]:
            conversations = example.get(conversations_column, [])

            # Convert to standard format
            messages = []
            for conv in conversations:
                if isinstance(conv, dict):
                    role = conv.get("from", conv.get("role", "user"))
                    content = conv.get("value", conv.get("content", ""))
                    messages.append({"role": role, "content": content})

            formatted = self.format_conversation(messages)

            if self.config.add_eos_token and formatted:
                formatted += "<|endoftext|>"

            example["formatted_text"] = formatted
            example["quality_score"] = self.estimate_quality(formatted)

            return example

        processed = dataset.map(process_conv)

        if self.config.check_quality:
            processed = processed.filter(
                lambda x: x.get("quality_score", 0) >= self.config.min_quality_score
            )

        return processed

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_count": self._processed_count,
            "filtered_count": self._filtered_count,
            "config": {
                "max_length": self.config.max_length,
                "min_length": self.config.min_length,
                "template": self.config.chat_template,
            },
        }

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._processed_count = 0
        self._filtered_count = 0
