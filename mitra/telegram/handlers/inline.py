"""
🤖 Mitra AI - Inline Query Handlers
Handles inline queries for inline mode.
Coded by Denvil with love 🤍
"""

from typing import List
from uuid import uuid4
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import ContextTypes

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class InlineHandlers:
    """Handlers for inline queries."""

    async def handle_inline(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming inline queries."""
        query = update.inline_query
        if not query:
            return

        user = update.effective_user
        text = query.query.strip()

        logger.info(
            "inline_query_received",
            user_id=user.id if user else None,
            query=text[:50],
        )

        if not text:
            # Show default suggestions
            results = self._get_default_suggestions()
        else:
            # Generate results based on query
            results = await self._generate_results(text)

        await query.answer(
            results=results[:50],  # Telegram limit
            cache_time=60,
            is_personal=True,
        )

    def _get_default_suggestions(self) -> List[InlineQueryResultArticle]:
        """Get default inline suggestions."""
        return [
            InlineQueryResultArticle(
                id=str(uuid4()),
                title="🤖 Ask Mitra AI",
                description="Type your question to get an AI response",
                input_message_content=InputTextMessageContent(
                    message_text="💭 Type your question after @MitraAI_bot",
                ),
            ),
            InlineQueryResultArticle(
                id=str(uuid4()),
                title="🧮 Calculate",
                description="Type a math expression to calculate",
                input_message_content=InputTextMessageContent(
                    message_text="🧮 Type a math expression after @MitraAI_bot calc",
                ),
            ),
            InlineQueryResultArticle(
                id=str(uuid4()),
                title="💻 Code Help",
                description="Get coding assistance",
                input_message_content=InputTextMessageContent(
                    message_text="💻 Type your coding question after @MitraAI_bot code",
                ),
            ),
        ]

    async def _generate_results(
        self,
        query: str,
    ) -> List[InlineQueryResultArticle]:
        """Generate results for a query."""
        results = []

        # Quick response option
        results.append(
            InlineQueryResultArticle(
                id=str(uuid4()),
                title=f"🤖 Ask: {query[:30]}{'...' if len(query) > 30 else ''}",
                description="Tap to get AI response",
                input_message_content=InputTextMessageContent(
                    message_text=(
                        f"🤖 *Mitra AI Response*\n\n"
                        f"Question: _{query}_\n\n"
                        f"Processing your request...\n\n"
                        f"_Coded by Denvil with love 🤍_"
                    ),
                    parse_mode="Markdown",
                ),
            )
        )

        # Check for special prefixes
        if query.lower().startswith("calc "):
            expression = query[5:].strip()
            results.insert(0, self._create_calc_result(expression))

        elif query.lower().startswith("code "):
            code_query = query[5:].strip()
            results.insert(0, self._create_code_result(code_query))

        return results

    def _create_calc_result(
        self,
        expression: str,
    ) -> InlineQueryResultArticle:
        """Create a calculation result."""
        try:
            # Safe evaluation (basic math only)
            # In production, use a proper math parser
            result = "Calculation feature is being configured"
        except Exception:
            result = "Invalid expression"

        return InlineQueryResultArticle(
            id=str(uuid4()),
            title=f"🧮 Calculate: {expression}",
            description=f"Result: {result}",
            input_message_content=InputTextMessageContent(
                message_text=f"🧮 {expression} = {result}\n\n_Coded by Denvil with love 🤍_",
                parse_mode="Markdown",
            ),
        )

    def _create_code_result(
        self,
        query: str,
    ) -> InlineQueryResultArticle:
        """Create a code help result."""
        return InlineQueryResultArticle(
            id=str(uuid4()),
            title=f"💻 Code: {query[:30]}{'...' if len(query) > 30 else ''}",
            description="Get coding assistance",
            input_message_content=InputTextMessageContent(
                message_text=(
                    f"💻 *Code Help Request*\n\n"
                    f"_{query}_\n\n"
                    f"Code assistance feature is being configured.\n\n"
                    f"_Coded by Denvil with love 🤍_"
                ),
                parse_mode="Markdown",
            ),
        )
