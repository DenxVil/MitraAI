"""
🤖 Mitra AI - Telegram Configuration
Comprehensive configuration for Telegram bot with webhook support.
Coded by Denvil with love 🤍
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TelegramConfig(BaseSettings):
    """Telegram bot configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="TELEGRAM_",
    )

    # Bot credentials
    bot_token: str = Field(default="", description="Telegram bot token from BotFather")
    bot_username: str = Field(default="", description="Bot username without @")

    # Webhook settings
    webhook_enabled: bool = Field(default=False, description="Enable webhook mode")
    webhook_url: str = Field(default="", description="Webhook URL (e.g., https://your-domain.com/webhook)")
    webhook_path: str = Field(default="/webhook", description="Webhook path")
    webhook_port: int = Field(default=8443, description="Webhook server port")
    webhook_secret: str = Field(default="", description="Webhook secret token for verification")
    webhook_certificate: str = Field(default="", description="Path to SSL certificate")
    webhook_private_key: str = Field(default="", description="Path to SSL private key")

    # Rate limiting
    rate_limit_messages_per_minute: int = Field(default=20, ge=1, le=100)
    rate_limit_commands_per_minute: int = Field(default=10, ge=1, le=50)
    rate_limit_media_per_minute: int = Field(default=5, ge=1, le=20)
    rate_limit_window_seconds: int = Field(default=60, ge=30, le=300)
    global_rate_limit_per_second: int = Field(default=30, ge=1, le=100)

    # Feature toggles
    enable_inline_mode: bool = Field(default=True, description="Enable inline queries")
    enable_payments: bool = Field(default=False, description="Enable payment handling")
    enable_web_app: bool = Field(default=False, description="Enable Telegram Web App")
    enable_voice_messages: bool = Field(default=True, description="Enable voice message processing")
    enable_image_generation: bool = Field(default=True, description="Enable AI image generation")
    enable_document_processing: bool = Field(default=True, description="Enable document analysis")

    # Message settings
    max_message_length: int = Field(default=4096, ge=100, le=10000)
    max_caption_length: int = Field(default=1024, ge=100, le=2048)
    typing_indicator_delay: float = Field(default=0.5, ge=0.1, le=5.0)
    streaming_chunk_size: int = Field(default=50, ge=10, le=200)

    # Admin settings
    admin_chat_ids: str = Field(default="", description="Comma-separated admin chat IDs")
    broadcast_enabled: bool = Field(default=True, description="Enable broadcast messages")
    maintenance_mode: bool = Field(default=False, description="Enable maintenance mode")

    # Logging
    log_messages: bool = Field(default=True, description="Log message content")
    log_user_actions: bool = Field(default=True, description="Log user actions")

    # Timeouts
    connect_timeout: float = Field(default=10.0, ge=1.0, le=60.0)
    read_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    write_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    pool_timeout: float = Field(default=10.0, ge=1.0, le=60.0)

    @property
    def admin_ids(self) -> Set[int]:
        """Parse admin chat IDs from comma-separated string."""
        if not self.admin_chat_ids:
            return set()
        try:
            return {int(id.strip()) for id in self.admin_chat_ids.split(",") if id.strip()}
        except ValueError:
            return set()

    @property
    def full_webhook_url(self) -> str:
        """Get full webhook URL including path."""
        if not self.webhook_url:
            return ""
        url = self.webhook_url.rstrip("/")
        path = self.webhook_path.lstrip("/")
        return f"{url}/{path}"

    def validate(self) -> None:
        """Validate required configuration."""
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        if self.webhook_enabled:
            if not self.webhook_url:
                raise ValueError("TELEGRAM_WEBHOOK_URL is required when webhook is enabled")


@dataclass
class WebhookSettings:
    """Webhook-specific settings."""
    url: str
    path: str = "/webhook"
    port: int = 8443
    secret_token: Optional[str] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    ip_address: Optional[str] = None
    max_connections: int = 40
    allowed_updates: Optional[List[str]] = None
    drop_pending_updates: bool = False


@dataclass
class RateLimitSettings:
    """Rate limiting configuration."""
    messages_per_minute: int = 20
    commands_per_minute: int = 10
    media_per_minute: int = 5
    window_seconds: int = 60
    global_per_second: int = 30


@dataclass
class FeatureFlags:
    """Feature toggles for the bot."""
    inline_mode: bool = True
    payments: bool = False
    web_app: bool = False
    voice_messages: bool = True
    image_generation: bool = True
    document_processing: bool = True
    streaming_responses: bool = True
    conversation_memory: bool = True


# Global config instance
telegram_config = TelegramConfig()
