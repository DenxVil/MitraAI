"""
🤖 Mitra AI - FastAPI Webhook Server
Production-ready webhook server for Telegram bot.
Coded by Denvil with love 🤍
"""

from typing import Optional, Callable, Any
from datetime import datetime, timezone
import asyncio
import hashlib
import hmac

try:
    from fastapi import FastAPI, Request, Response, HTTPException
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    Request = None
    Response = None
    HTTPException = None
    JSONResponse = None

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import TelegramConfig, telegram_config


class WebhookServer:
    """
    FastAPI-based webhook server for Telegram bot.

    Features:
    - Secure webhook handling with secret token verification
    - Health check endpoints
    - Request logging
    - Graceful shutdown
    """

    def __init__(
        self,
        config: Optional[TelegramConfig] = None,
        update_handler: Optional[Callable] = None,
    ) -> None:
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for webhook server. "
                "Install with: pip install fastapi uvicorn"
            )

        self.config = config or telegram_config
        self.update_handler = update_handler
        self._app: Optional[FastAPI] = None
        self._server = None
        self._start_time: Optional[datetime] = None

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="Mitra AI Webhook Server",
            description="Telegram bot webhook endpoint",
            version="1.0.0",
            docs_url=None,  # Disable docs in production
            redoc_url=None,
        )

        # Health check endpoint
        @app.get("/health")
        async def health_check() -> dict:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime": self._get_uptime(),
            }

        # Readiness check endpoint
        @app.get("/ready")
        async def readiness_check() -> dict:
            """Readiness check endpoint."""
            return {
                "ready": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Webhook endpoint
        @app.post(self.config.webhook_path)
        async def webhook(request: Request) -> Response:
            """Handle incoming webhook updates."""
            try:
                # Verify secret token if configured
                if self.config.webhook_secret:
                    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
                    if not self._verify_secret(secret_token):
                        logger.warning("webhook_invalid_secret")
                        raise HTTPException(status_code=403, detail="Invalid secret token")

                # Parse update
                update_data = await request.json()

                logger.info(
                    "webhook_update_received",
                    update_id=update_data.get("update_id"),
                )

                # Process update
                if self.update_handler:
                    asyncio.create_task(
                        self._process_update(update_data)
                    )

                return Response(status_code=200)

            except HTTPException:
                raise
            except Exception as e:
                logger.error("webhook_error", error=str(e))
                return Response(status_code=200)  # Always return 200 to Telegram

        # Startup event
        @app.on_event("startup")
        async def startup() -> None:
            """Application startup."""
            self._start_time = datetime.now(timezone.utc)
            logger.info("webhook_server_started")

        # Shutdown event
        @app.on_event("shutdown")
        async def shutdown() -> None:
            """Application shutdown."""
            logger.info("webhook_server_stopped")

        self._app = app
        return app

    def _verify_secret(self, provided_secret: Optional[str]) -> bool:
        """Verify the secret token from Telegram."""
        if not self.config.webhook_secret:
            return True

        if not provided_secret:
            return False

        return hmac.compare_digest(
            provided_secret.encode(),
            self.config.webhook_secret.encode(),
        )

    async def _process_update(self, update_data: dict) -> None:
        """Process an update from Telegram."""
        try:
            if self.update_handler:
                await self.update_handler(update_data)
        except Exception as e:
            logger.error("update_processing_error", error=str(e))

    def _get_uptime(self) -> str:
        """Get server uptime."""
        if not self._start_time:
            return "0s"

        delta = datetime.now(timezone.utc) - self._start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    async def run(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
    ) -> None:
        """Run the webhook server."""
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to run the webhook server. "
                "Install with: pip install uvicorn"
            )

        app = self.create_app()
        port = port or self.config.webhook_port

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        self._server = server

        logger.info(
            "webhook_server_starting",
            host=host,
            port=port,
            path=self.config.webhook_path,
        )

        await server.serve()

    def stop(self) -> None:
        """Stop the webhook server."""
        if self._server:
            self._server.should_exit = True
            logger.info("webhook_server_stopping")

    @property
    def app(self) -> Optional[FastAPI]:
        """Get the FastAPI application."""
        return self._app
