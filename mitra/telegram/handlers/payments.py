"""
🤖 Mitra AI - Payment Handlers
Handles Telegram payment processing.
Coded by Denvil with love 🤍
"""

from typing import Optional
from telegram import Update, LabeledPrice
from telegram.ext import ContextTypes

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class PaymentHandlers:
    """Handlers for payment-related operations."""

    async def handle_pre_checkout(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle pre-checkout queries."""
        query = update.pre_checkout_query
        if not query:
            return

        logger.info(
            "pre_checkout_query",
            user_id=query.from_user.id,
            invoice_payload=query.invoice_payload,
            total_amount=query.total_amount,
            currency=query.currency,
        )

        # Validate the payment
        # In production, verify inventory, check user eligibility, etc.
        try:
            # Accept the payment
            await query.answer(ok=True)
        except Exception as e:
            logger.error("pre_checkout_error", error=str(e))
            await query.answer(
                ok=False,
                error_message="Payment validation failed. Please try again.",
            )

    async def handle_successful_payment(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle successful payments."""
        message = update.message
        if not message or not message.successful_payment:
            return

        payment = message.successful_payment
        user = update.effective_user

        logger.info(
            "successful_payment",
            user_id=user.id if user else None,
            total_amount=payment.total_amount,
            currency=payment.currency,
            invoice_payload=payment.invoice_payload,
            provider_payment_charge_id=payment.provider_payment_charge_id,
        )

        # Process the payment
        await self._process_payment(payment, user, context)

        # Thank the user
        await context.bot.send_message(
            chat_id=message.chat_id,
            text=(
                "✅ *Payment Successful!*\n\n"
                f"Amount: {payment.total_amount / 100} {payment.currency}\n"
                f"Thank you for your purchase!\n\n"
                "_Coded by Denvil with love 🤍_"
            ),
            parse_mode="Markdown",
        )

    async def _process_payment(
        self,
        payment,
        user,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Process a successful payment."""
        # Implement payment processing logic
        # - Update user subscription
        # - Grant premium features
        # - Store payment record
        pass

    async def create_invoice(
        self,
        chat_id: int,
        title: str,
        description: str,
        payload: str,
        provider_token: str,
        currency: str,
        prices: list,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Create and send an invoice."""
        labeled_prices = [
            LabeledPrice(label=p["label"], amount=p["amount"])
            for p in prices
        ]

        await context.bot.send_invoice(
            chat_id=chat_id,
            title=title,
            description=description,
            payload=payload,
            provider_token=provider_token,
            currency=currency,
            prices=labeled_prices,
        )

        logger.info(
            "invoice_sent",
            chat_id=chat_id,
            title=title,
            currency=currency,
        )
