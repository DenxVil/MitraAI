"""
🤖 Mitra AI - Message Handlers
Handles text message processing.
Coded by Denvil with love 🤍
"""

from datetime import datetime, timezone
from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MessageHandlers:
    """Handlers for text messages."""

    async def handle_text(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming text messages."""
        message = update.message
        if not message or not message.text:
            return

        user = update.effective_user
        if not user:
            return

        text = message.text
        chat_id = message.chat_id

        # Update user stats
        context.user_data["message_count"] = context.user_data.get("message_count", 0) + 1
        if "first_interaction" not in context.user_data:
            context.user_data["first_interaction"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "text_message_received",
            user_id=user.id,
            chat_id=chat_id,
            message_length=len(text),
        )

        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        try:
            # Process message through AI engine
            response = await self._process_with_ai(text, user.id, context)

            # Send response
            await context.bot.send_message(
                chat_id=chat_id,
                text=response,
                parse_mode="Markdown",
            )

        except Exception as e:
            logger.error(
                "message_processing_error",
                user_id=user.id,
                error=str(e),
            )
            await context.bot.send_message(
                chat_id=chat_id,
                text="I encountered an error processing your message. Please try again.",
            )

    async def _process_with_ai(
        self,
        text: str,
        user_id: int,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> str:
        """Process text through AI engine."""
        try:
            # Get or create conversation context
            conversation = context.user_data.get("conversation", [])

            # Add user message to conversation
            conversation.append({
                "role": "user",
                "content": text,
            })

            # Keep conversation limited
            max_messages = 20
            if len(conversation) > max_messages:
                conversation = conversation[-max_messages:]

            # Generate response (placeholder - connect to actual AI)
            response = await self._generate_response(text, conversation)

            # Add assistant response to conversation
            conversation.append({
                "role": "assistant",
                "content": response,
            })

            # Save conversation
            context.user_data["conversation"] = conversation

            return response

        except Exception as e:
            logger.error("ai_processing_error", error=str(e))
            raise

    async def _generate_response(
        self,
        text: str,
        conversation: list,
    ) -> str:
        """Generate AI response for the text."""
        # This is a placeholder - connect to MitraSuperBrain
        # In production, this would call the superintelligence module

        # For now, return a helpful message
        return (
            "I'm Mitra AI, your superintelligent assistant! 🤖\n\n"
            f"You said: *{text[:100]}{'...' if len(text) > 100 else ''}*\n\n"
            "I'm being set up to provide intelligent responses. "
            "The AI engine is being configured.\n\n"
            "_Coded by Denvil with love 🤍_"
        )
