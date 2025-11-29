"""
🤖 Mitra AI - Main Bot Orchestrator
Advanced Telegram bot with middleware pipeline and streaming responses.
Coded by Denvil with love 🤍
"""

import asyncio
import signal
from typing import Optional, List, Callable, Any, Dict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from telegram import Update, Bot
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    InlineQueryHandler,
    filters,
)

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .config import TelegramConfig, telegram_config
from .handlers import (
    CommandHandlers,
    MessageHandlers,
    MediaHandlers,
    CallbackHandlers,
    InlineHandlers,
)
from .middleware import (
    AuthMiddleware,
    RateLimitMiddleware,
    LoggingMiddleware,
)


class MiddlewarePipeline:
    """Pipeline for processing middleware in order."""

    def __init__(self) -> None:
        self._middlewares: List[Callable] = []

    def add(self, middleware: Callable) -> "MiddlewarePipeline":
        """Add a middleware to the pipeline."""
        self._middlewares.append(middleware)
        return self

    async def process(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """
        Process update through all middlewares.

        Returns:
            True if update should continue processing, False to stop.
        """
        for middleware in self._middlewares:
            try:
                result = await middleware(update, context)
                if result is False:
                    return False
            except Exception as e:
                logger.error(
                    "middleware_error",
                    middleware=middleware.__name__,
                    error=str(e),
                )
                # Continue processing despite middleware error
        return True


class MitraBot:
    """
    Main Telegram bot orchestrator.

    Features:
    - Middleware pipeline for pre-processing
    - Streaming responses for long content
    - Graceful shutdown handling
    - Admin panel integration
    - Comprehensive error handling
    """

    def __init__(
        self,
        config: Optional[TelegramConfig] = None,
    ) -> None:
        self.config = config or telegram_config
        self.config.validate()

        self._application: Optional[Application] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Initialize middleware pipeline
        self._middleware = MiddlewarePipeline()
        self._setup_middleware()

        # Initialize handlers
        self._command_handlers = CommandHandlers()
        self._message_handlers = MessageHandlers()
        self._media_handlers = MediaHandlers()
        self._callback_handlers = CallbackHandlers()
        self._inline_handlers = InlineHandlers()

        logger.info(
            "mitra_bot_initialized",
            webhook_enabled=self.config.webhook_enabled,
            inline_enabled=self.config.enable_inline_mode,
        )

    def _setup_middleware(self) -> None:
        """Configure the middleware pipeline."""
        # Logging middleware first
        self._middleware.add(LoggingMiddleware(self.config).process)

        # Authentication middleware
        self._middleware.add(AuthMiddleware(self.config).process)

        # Rate limiting middleware
        self._middleware.add(RateLimitMiddleware(self.config).process)

    async def _pre_process(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """Pre-process update through middleware pipeline."""
        return await self._middleware.process(update, context)

    def build_application(self) -> Application:
        """Build the Telegram application with all handlers."""
        builder = ApplicationBuilder()
        builder.token(self.config.bot_token)
        builder.connect_timeout(self.config.connect_timeout)
        builder.read_timeout(self.config.read_timeout)
        builder.write_timeout(self.config.write_timeout)
        builder.pool_timeout(self.config.pool_timeout)

        app = builder.build()

        # Add command handlers
        app.add_handler(CommandHandler("start", self._wrap_handler(self._command_handlers.start)))
        app.add_handler(CommandHandler("help", self._wrap_handler(self._command_handlers.help)))
        app.add_handler(CommandHandler("settings", self._wrap_handler(self._command_handlers.settings)))
        app.add_handler(CommandHandler("clear", self._wrap_handler(self._command_handlers.clear)))
        app.add_handler(CommandHandler("stats", self._wrap_handler(self._command_handlers.stats)))
        app.add_handler(CommandHandler("admin", self._wrap_handler(self._command_handlers.admin)))

        # Add message handlers
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._wrap_handler(self._message_handlers.handle_text)
        ))

        # Add media handlers
        app.add_handler(MessageHandler(
            filters.PHOTO,
            self._wrap_handler(self._media_handlers.handle_photo)
        ))
        app.add_handler(MessageHandler(
            filters.VOICE,
            self._wrap_handler(self._media_handlers.handle_voice)
        ))
        app.add_handler(MessageHandler(
            filters.Document.ALL,
            self._wrap_handler(self._media_handlers.handle_document)
        ))

        # Add callback handlers
        app.add_handler(CallbackQueryHandler(
            self._wrap_handler(self._callback_handlers.handle_callback)
        ))

        # Add inline handlers if enabled
        if self.config.enable_inline_mode:
            app.add_handler(InlineQueryHandler(
                self._wrap_handler(self._inline_handlers.handle_inline)
            ))

        # Add error handler
        app.add_error_handler(self._error_handler)

        self._application = app
        return app

    def _wrap_handler(self, handler: Callable) -> Callable:
        """Wrap handler with middleware pre-processing."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            # Run through middleware pipeline
            should_continue = await self._pre_process(update, context)
            if not should_continue:
                return

            # Execute handler
            await handler(update, context)

        return wrapped

    async def _error_handler(
        self,
        update: object,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle errors during update processing."""
        logger.error(
            "telegram_error",
            error=str(context.error),
            update=str(update) if update else None,
        )

        # Try to notify user of error
        if isinstance(update, Update) and update.effective_chat:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="I encountered an error processing your request. Please try again.",
                )
            except Exception as e:
                logger.error("error_notification_failed", error=str(e))

    async def send_streaming_response(
        self,
        chat_id: int,
        content_generator: Any,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """
        Send a streaming response with live updates.

        Args:
            chat_id: Chat to send message to
            content_generator: Async generator yielding content chunks
            context: Telegram context
        """
        message = None
        content = ""
        last_update = datetime.now(timezone.utc)
        update_interval = 0.5  # Minimum seconds between edits

        try:
            # Send initial typing indicator
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")

            async for chunk in content_generator:
                content += chunk

                # Update message periodically
                now = datetime.now(timezone.utc)
                if (now - last_update).total_seconds() >= update_interval:
                    if message is None:
                        message = await context.bot.send_message(
                            chat_id=chat_id,
                            text=content + "▌",  # Cursor indicator
                        )
                    else:
                        try:
                            await message.edit_text(content + "▌")
                        except Exception:
                            pass  # Ignore edit errors
                    last_update = now

            # Final update without cursor
            if message:
                await message.edit_text(content)
            else:
                await context.bot.send_message(chat_id=chat_id, text=content)

        except Exception as e:
            logger.error("streaming_response_error", error=str(e))
            if message is None:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Sorry, I encountered an error generating the response.",
                )

    async def start(self) -> None:
        """Start the bot."""
        if self._running:
            logger.warning("bot_already_running")
            return

        app = self.build_application()
        self._running = True

        logger.info("mitra_bot_starting")

        if self.config.webhook_enabled:
            await self._start_webhook(app)
        else:
            await self._start_polling(app)

    async def _start_polling(self, app: Application) -> None:
        """Start bot in polling mode."""
        logger.info("starting_polling_mode")

        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        await self._shutdown(app)

    async def _start_webhook(self, app: Application) -> None:
        """Start bot in webhook mode."""
        logger.info(
            "starting_webhook_mode",
            url=self.config.full_webhook_url,
            port=self.config.webhook_port,
        )

        await app.initialize()
        await app.start()

        await app.bot.set_webhook(
            url=self.config.full_webhook_url,
            secret_token=self.config.webhook_secret or None,
            drop_pending_updates=True,
        )

        # Start webhook server
        await app.updater.start_webhook(
            listen="0.0.0.0",
            port=self.config.webhook_port,
            url_path=self.config.webhook_path,
            secret_token=self.config.webhook_secret or None,
        )

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        await self._shutdown(app)

    async def _shutdown(self, app: Application) -> None:
        """Graceful shutdown."""
        logger.info("mitra_bot_shutting_down")

        if self.config.webhook_enabled:
            await app.bot.delete_webhook()

        await app.updater.stop()
        await app.stop()
        await app.shutdown()

        self._running = False
        logger.info("mitra_bot_stopped")

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        logger.info("shutdown_requested")
        self._shutdown_event.set()

    def run(self) -> None:
        """Run the bot (blocking)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.request_shutdown)

        try:
            loop.run_until_complete(self.start())
        finally:
            loop.close()

    @asynccontextmanager
    async def session(self):
        """Async context manager for bot session."""
        try:
            await self.start()
            yield self
        finally:
            self.request_shutdown()
