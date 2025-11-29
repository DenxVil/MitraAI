"""
🤖 Mitra AI - Command Handlers
Handles Telegram bot commands.
Coded by Denvil with love 🤍
"""

from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class CommandHandlers:
    """Handlers for bot commands."""

    async def start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /start command."""
        user = update.effective_user
        if not user or not update.effective_chat:
            return

        welcome_message = f"""
🤖 *Welcome to Mitra AI!*

Hello, {user.first_name}! I'm Mitra, your superintelligent AI assistant.

I can help you with:
• 💬 Natural conversations
• 🧮 Mathematics and calculations
• 💻 Coding and programming
• 🔍 Reasoning and analysis
• 🎨 Creative tasks

*Commands:*
/help - Show help message
/settings - Bot settings
/clear - Clear conversation
/stats - Your statistics

Just send me a message to get started!

_Coded by Denvil with love 🤍_
        """

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=welcome_message,
            parse_mode="Markdown",
        )

        logger.info(
            "start_command",
            user_id=user.id,
            username=user.username,
        )

    async def help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /help command."""
        if not update.effective_chat:
            return

        help_message = """
🆘 *Mitra AI Help*

*Basic Commands:*
/start - Start the bot
/help - Show this help message
/clear - Clear conversation history
/settings - Adjust your settings
/stats - View your statistics

*Features:*
• Send text messages for AI responses
• Send images for analysis
• Send voice messages for transcription
• Send documents for processing

*Tips:*
• Be specific in your questions
• Use context from previous messages
• Try different thinking modes

*Admin Commands:* (for admins only)
/admin - Access admin panel

Need more help? Just ask me anything!

_Coded by Denvil with love 🤍_
        """

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=help_message,
            parse_mode="Markdown",
        )

    async def settings(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /settings command."""
        if not update.effective_chat:
            return

        from ..keyboards import SettingsKeyboard

        keyboard = SettingsKeyboard.main_menu()

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="⚙️ *Settings*\n\nChoose what you'd like to configure:",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )

    async def clear(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /clear command - clears conversation history."""
        user = update.effective_user
        if not user or not update.effective_chat:
            return

        # Clear user's conversation from context
        if "conversation" in context.user_data:
            del context.user_data["conversation"]

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="🗑️ Conversation history cleared! Let's start fresh.",
        )

        logger.info("clear_command", user_id=user.id)

    async def stats(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /stats command - show user statistics."""
        user = update.effective_user
        if not user or not update.effective_chat:
            return

        # Get user stats from context
        message_count = context.user_data.get("message_count", 0)
        first_interaction = context.user_data.get("first_interaction", "N/A")

        stats_message = f"""
📊 *Your Statistics*

👤 *User:* {user.first_name}
🆔 *ID:* `{user.id}`
💬 *Messages:* {message_count}
📅 *First interaction:* {first_interaction}

_Coded by Denvil with love 🤍_
        """

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=stats_message,
            parse_mode="Markdown",
        )

    async def admin(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /admin command - access admin panel."""
        user = update.effective_user
        if not user or not update.effective_chat:
            return

        from ..admin import AdminPanel

        panel = AdminPanel()

        # Check if user is admin
        if not await panel.is_admin(user.id):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⛔ You don't have admin permissions.",
            )
            return

        # Show admin panel
        keyboard = await panel.get_main_menu(user.id)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="🔐 *Admin Panel*\n\nChoose an option:",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )

        logger.info("admin_command", user_id=user.id)
