"""
🤖 Mitra AI - Callback Handlers
Handles inline button callbacks.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any
from telegram import Update, CallbackQuery
from telegram.ext import ContextTypes

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class CallbackHandlers:
    """Handlers for callback queries from inline buttons."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Any] = {
            "settings": self._handle_settings,
            "admin": self._handle_admin,
            "thinking_mode": self._handle_thinking_mode,
            "confirm": self._handle_confirmation,
            "cancel": self._handle_cancel,
        }

    async def handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Route callback queries to appropriate handlers."""
        query = update.callback_query
        if not query or not query.data:
            return

        user = update.effective_user
        if not user:
            return

        # Acknowledge the callback
        await query.answer()

        data = query.data
        logger.info(
            "callback_received",
            user_id=user.id,
            callback_data=data,
        )

        # Parse callback data (format: action:params)
        parts = data.split(":", 1)
        action = parts[0]
        params = parts[1] if len(parts) > 1 else ""

        # Route to handler
        handler = self._handlers.get(action)
        if handler:
            await handler(query, context, params)
        else:
            logger.warning(
                "unknown_callback",
                action=action,
                data=data,
            )

    async def _handle_settings(
        self,
        query: CallbackQuery,
        context: ContextTypes.DEFAULT_TYPE,
        params: str,
    ) -> None:
        """Handle settings-related callbacks."""
        if params == "main":
            from ..keyboards import SettingsKeyboard
            keyboard = SettingsKeyboard.main_menu()
            await query.edit_message_text(
                text="⚙️ *Settings*\n\nChoose what you'd like to configure:",
                parse_mode="Markdown",
                reply_markup=keyboard,
            )

        elif params == "notifications":
            # Toggle notifications
            current = context.user_data.get("notifications_enabled", True)
            context.user_data["notifications_enabled"] = not current
            status = "enabled" if not current else "disabled"
            await query.edit_message_text(
                text=f"🔔 Notifications {status}!",
            )

        elif params == "language":
            # Show language options
            await query.edit_message_text(
                text="🌐 *Language Settings*\n\nLanguage selection coming soon!",
                parse_mode="Markdown",
            )

    async def _handle_admin(
        self,
        query: CallbackQuery,
        context: ContextTypes.DEFAULT_TYPE,
        params: str,
    ) -> None:
        """Handle admin panel callbacks."""
        from ..admin import AdminPanel

        panel = AdminPanel()
        user = query.from_user

        if not await panel.is_admin(user.id):
            await query.edit_message_text("⛔ Admin access denied.")
            return

        if params == "dashboard":
            stats = await panel.get_dashboard_stats()
            await query.edit_message_text(
                text=f"📊 *Dashboard*\n\n{stats}",
                parse_mode="Markdown",
            )

        elif params == "users":
            # Show user management
            await query.edit_message_text(
                text="👥 *User Management*\n\nUser management coming soon!",
                parse_mode="Markdown",
            )

        elif params == "ai_control":
            # Show AI control options
            await query.edit_message_text(
                text="🧠 *AI Control*\n\nAI control panel coming soon!",
                parse_mode="Markdown",
            )

        elif params == "broadcast":
            # Show broadcast options
            await query.edit_message_text(
                text="📢 *Broadcast*\n\nBroadcast feature coming soon!",
                parse_mode="Markdown",
            )

    async def _handle_thinking_mode(
        self,
        query: CallbackQuery,
        context: ContextTypes.DEFAULT_TYPE,
        params: str,
    ) -> None:
        """Handle thinking mode selection."""
        modes = {
            "instant": "⚡ Instant - Quick responses",
            "standard": "🔵 Standard - Balanced thinking",
            "deep": "🧠 Deep - Thorough analysis",
            "expert": "🎓 Expert - Domain expertise",
            "maximum": "🔮 Maximum - All capabilities",
        }

        if params in modes:
            context.user_data["thinking_mode"] = params
            await query.edit_message_text(
                text=f"Thinking mode set to: {modes[params]}",
            )
        else:
            from ..keyboards import ThinkingModeKeyboard
            keyboard = ThinkingModeKeyboard.get_keyboard()
            await query.edit_message_text(
                text="🧠 *Select Thinking Mode:*",
                parse_mode="Markdown",
                reply_markup=keyboard,
            )

    async def _handle_confirmation(
        self,
        query: CallbackQuery,
        context: ContextTypes.DEFAULT_TYPE,
        params: str,
    ) -> None:
        """Handle confirmation callbacks."""
        pending_action = context.user_data.get("pending_action")
        if not pending_action:
            await query.edit_message_text("No pending action to confirm.")
            return

        # Execute pending action
        action_type = pending_action.get("type")
        if action_type == "clear_conversation":
            if "conversation" in context.user_data:
                del context.user_data["conversation"]
            await query.edit_message_text("✅ Conversation cleared!")

        elif action_type == "delete_data":
            context.user_data.clear()
            await query.edit_message_text("✅ All your data has been deleted!")

        # Clear pending action
        del context.user_data["pending_action"]

    async def _handle_cancel(
        self,
        query: CallbackQuery,
        context: ContextTypes.DEFAULT_TYPE,
        params: str,
    ) -> None:
        """Handle cancel callbacks."""
        if "pending_action" in context.user_data:
            del context.user_data["pending_action"]

        await query.edit_message_text("❌ Action cancelled.")
