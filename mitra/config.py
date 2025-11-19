"""
Configuration management for Mitra AI.

Handles environment variables, settings validation, and configuration loading.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # Local AI Model Configuration
    local_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    local_model_device: str = "auto"
    local_model_quantize: bool = True
    local_model_max_tokens: int = 512

    # Telegram Bot
    telegram_bot_token: str = ""

    # Application Behavior
    max_conversation_history: int = Field(default=10, ge=1, le=50)
    rate_limit_messages_per_minute: int = Field(default=20, ge=1, le=100)
    max_message_length: int = Field(default=4000, ge=100, le=10000)

    # Safety & Moderation
    enable_content_moderation: bool = True

    # Azure Application Insights
    applicationinsights_connection_string: str = ""

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def validate_required_settings(self) -> None:
        """Validate that all required settings are present."""
        if not self.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        # Local model is always used, no API keys needed


# Global settings instance
settings = Settings()
