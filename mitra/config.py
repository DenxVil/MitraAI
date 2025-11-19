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
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # Azure OpenAI Configuration
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment_name: str = "gpt-4"
    azure_openai_api_version: str = "2024-02-15-preview"

    # Alternative OpenAI Configuration
    openai_api_key: str = ""

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

    @property
    def use_azure_openai(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    def validate_required_settings(self) -> None:
        """Validate that all required settings are present."""
        if not self.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        if not self.use_azure_openai and not self.openai_api_key:
            raise ValueError(
                "Either Azure OpenAI credentials or OpenAI API key must be provided"
            )


# Global settings instance
settings = Settings()
