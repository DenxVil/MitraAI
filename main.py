"""
Main entry point for Mitra AI.

Initializes and runs the Mitra Telegram bot.
"""

import asyncio
import sys
from mitra.config import settings
from mitra.utils import setup_logging, get_logger
from mitra.bot import MitraTelegramBot


def main() -> None:
    """Main application entry point."""
    # Setup logging
    setup_logging(log_level=settings.log_level, environment=settings.environment)

    logger = get_logger(__name__)

    logger.info("mitra_starting", environment=settings.environment, version="0.1.0")

    try:
        # Validate settings
        settings.validate_required_settings()

        # Create and run bot
        bot = MitraTelegramBot()

        logger.info("mitra_ready")
        print("🤖 Mitra AI is now running...")
        print(f"Environment: {settings.environment}")
        print(
            f"Model: {settings.azure_openai_deployment_name if settings.use_azure_openai else 'OpenAI GPT-4'}"
        )
        print("\nPress Ctrl+C to stop.\n")

        bot.run()

    except KeyboardInterrupt:
        logger.info("mitra_shutting_down_gracefully")
        print("\n👋 Mitra AI shutting down...")
        sys.exit(0)

    except Exception as e:
        logger.error("mitra_startup_failed", error=str(e), error_type=type(e).__name__)
        print(f"\n❌ Failed to start Mitra AI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
