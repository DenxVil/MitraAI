"""
🤖 Mitra AI - Main Entry Point
Superintelligent AI system with Telegram interface.
Coded by Denvil with love 🤍
"""

import asyncio
import sys
from typing import Optional

try:
    import click
except ImportError:
    click = None

from mitra.config import settings
from mitra.utils import setup_logging, get_logger


def main() -> None:
    """Main application entry point."""
    # Setup logging
    setup_logging(log_level=settings.log_level, environment=settings.environment)

    logger = get_logger(__name__)

    logger.info("mitra_starting", environment=settings.environment, version="1.0.0")

    try:
        # Validate settings
        settings.validate_required_settings()

        # Create and run bot
        from mitra.bot import MitraTelegramBot
        bot = MitraTelegramBot()

        logger.info("mitra_ready")
        print("🤖 Mitra AI is now running...")
        print(f"Environment: {settings.environment}")
        print(
            f"Model: {settings.azure_openai_deployment_name if settings.use_azure_openai else 'Local Model'}"
        )
        print("\nPress Ctrl+C to stop.\n")
        print("_Coded by Denvil with love 🤍_\n")

        bot.run()

    except KeyboardInterrupt:
        logger.info("mitra_shutting_down_gracefully")
        print("\n👋 Mitra AI shutting down...")
        sys.exit(0)

    except Exception as e:
        logger.error("mitra_startup_failed", error=str(e), error_type=type(e).__name__)
        print(f"\n❌ Failed to start Mitra AI: {e}")
        sys.exit(1)


def run_telegram_advanced() -> None:
    """Run the advanced Telegram bot with all features."""
    setup_logging(log_level=settings.log_level, environment=settings.environment)

    logger = get_logger(__name__)
    logger.info("mitra_advanced_starting", version="1.0.0")

    try:
        from mitra.telegram import MitraBot, TelegramConfig

        config = TelegramConfig()
        bot = MitraBot(config)

        print("🤖 Mitra AI Advanced Mode")
        print("=" * 40)
        print(f"Webhook: {'Enabled' if config.webhook_enabled else 'Polling'}")
        print(f"Inline Mode: {'Enabled' if config.enable_inline_mode else 'Disabled'}")
        print("=" * 40)
        print("\n_Coded by Denvil with love 🤍_\n")

        bot.run()

    except KeyboardInterrupt:
        print("\n👋 Mitra AI shutting down...")
        sys.exit(0)

    except Exception as e:
        logger.error("mitra_advanced_failed", error=str(e))
        print(f"\n❌ Failed to start: {e}")
        sys.exit(1)


async def run_ai_demo() -> None:
    """Run a demo of the AI superintelligence."""
    print("🧠 Mitra AI Superintelligence Demo")
    print("=" * 50)

    try:
        from mitra.ai.superintelligence import MitraSuperBrain, ThinkingMode

        brain = MitraSuperBrain()

        print("\nInitializing MitraSuperBrain...")
        print("Note: Model loading may take a few minutes on first run.\n")

        # Demo questions
        questions = [
            ("What is 15 + 27?", ThinkingMode.INSTANT),
            ("Explain why the sky is blue.", ThinkingMode.STANDARD),
            ("Solve: If x + 5 = 12, what is x?", ThinkingMode.DEEP),
        ]

        for question, mode in questions:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            print(f"Mode: {mode.name}")
            print("-" * 50)

            result = await brain.think(question, mode=mode)

            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.final_confidence:.1%}")
            print(f"Time: {result.thinking_time_ms:.0f}ms")

        print("\n" + "=" * 50)
        print("Demo complete!")
        print("\n_Coded by Denvil with love 🤍_")

    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Make sure AI dependencies are installed:")
        print("pip install torch transformers accelerate bitsandbytes")


if click:
    @click.group()
    @click.version_option(version="1.0.0", prog_name="Mitra AI")
    def cli():
        """🤖 Mitra AI - Superintelligent Assistant - Coded by Denvil with love 🤍"""
        pass

    @cli.command()
    def bot():
        """Run the basic Telegram bot."""
        main()

    @cli.command()
    def advanced():
        """Run the advanced Telegram bot with all features."""
        run_telegram_advanced()

    @cli.command()
    def demo():
        """Run an AI demonstration."""
        asyncio.run(run_ai_demo())

    @cli.command()
    @click.option("--host", default="0.0.0.0", help="Host to bind to")
    @click.option("--port", default=8443, type=int, help="Port to listen on")
    def webhook(host: str, port: int):
        """Run the webhook server."""
        from mitra.telegram.webhook import WebhookServer

        print(f"🌐 Starting webhook server on {host}:{port}")
        server = WebhookServer()
        asyncio.run(server.run(host=host, port=port))

    @cli.command()
    def info():
        """Show system information."""
        print("🤖 Mitra AI System Information")
        print("=" * 40)
        print(f"Version: 1.0.0")
        print(f"Environment: {settings.environment}")
        print(f"Log Level: {settings.log_level}")
        print("\nComponents:")
        print("  ✅ Core Error Handling")
        print("  ✅ Telegram Interface")
        print("  ✅ Admin Panel")
        print("  ✅ AI Superintelligence")
        print("  ✅ Training System")
        print("  ✅ Benchmark Suite")
        print("=" * 40)
        print("\n_Coded by Denvil with love 🤍_")


if __name__ == "__main__":
    if click and len(sys.argv) > 1:
        cli()
    else:
        main()
