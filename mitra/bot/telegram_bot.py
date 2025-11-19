"""
Telegram bot implementation for Mitra AI.

Handles Telegram interactions, commands, and message routing.
"""

from typing import Optional
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.error import TelegramError

from ..core import MitraEngine
from ..config import settings
from ..models import User, UserProfile
from ..utils import get_logger, ErrorHandler, RateLimiter, add_correlation_id
from uuid import uuid4


logger = get_logger(__name__)


class MitraTelegramBot:
    """
    Telegram bot interface for Mitra AI.
    
    Manages user interactions via Telegram and routes to core engine.
    """
    
    def __init__(self):
        """Initialize the Telegram bot."""
        self.engine = MitraEngine()
        self.rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_messages_per_minute,
            window_seconds=60
        )
        
        # User storage (in-memory for now)
        self.users: dict[str, User] = {}
        
        # Build application
        self.app = Application.builder().token(settings.telegram_bot_token).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("telegram_bot_initialized")
    
    def _register_handlers(self) -> None:
        """Register command and message handlers."""
        # Commands
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("clear", self.clear_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        
        # Message handler
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        
        # Error handler
        self.app.add_error_handler(self.error_handler)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        correlation_id = str(uuid4())
        add_correlation_id(correlation_id)
        
        user = update.effective_user
        logger.info(
            "start_command_received",
            user_id=user.id,
            username=user.username
        )
        
        # Create or get user
        self._get_or_create_user(user)
        
        welcome_message = """👋 Hello! I'm Mitra, your emotionally intelligent AI assistant.

I'm here to:
✨ Listen and understand you deeply
🧠 Help you think through complex problems
💙 Provide emotional support when you need it
🎯 Assist with decision-making and planning

I combine deep reasoning with emotional awareness to support you in the best way possible.

**How to use:**
- Just send me a message and I'll respond
- Use /help to see available commands
- Use /clear to start a fresh conversation

What's on your mind today?"""
        
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        correlation_id = str(uuid4())
        add_correlation_id(correlation_id)
        
        help_text = """🤖 **Mitra AI - Help**

**Available Commands:**
/start - Start a conversation with Mitra
/help - Show this help message
/clear - Clear conversation history and start fresh
/status - Show your usage statistics

**How I work:**
- Send me any message and I'll thoughtfully respond
- I analyze emotions and adapt my responses
- I maintain conversation context
- I prioritize your safety and wellbeing

**What I can help with:**
- Problem-solving and decision-making
- Emotional support and understanding
- Planning and organizing thoughts
- Learning and explaining concepts
- General conversation and companionship

**Important:**
- I'm an AI assistant, not a human
- For crises, please contact professionals (use /crisis for resources)
- I respect your privacy and boundaries

Just start chatting - I'm here to help! 💙"""
        
        await update.message.reply_text(help_text)
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command to reset conversation."""
        correlation_id = str(uuid4())
        add_correlation_id(correlation_id)
        
        user_id = str(update.effective_user.id)
        
        # Find and clear user's active conversation
        # (simplified - in production, track conversation IDs per user)
        cleared = False
        for conv_id, conv in list(self.engine.conversations.items()):
            if conv.user_id == user_id:
                self.engine.clear_conversation(conv_id)
                cleared = True
        
        if cleared:
            message = "✨ Conversation cleared! Let's start fresh. What would you like to talk about?"
        else:
            message = "No active conversation to clear. Feel free to start chatting!"
        
        await update.message.reply_text(message)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command to show usage stats."""
        correlation_id = str(uuid4())
        add_correlation_id(correlation_id)
        
        user_id = str(update.effective_user.id)
        user = self.users.get(user_id)
        
        if not user:
            await update.message.reply_text("No usage data available yet. Start chatting!")
            return
        
        # Get rate limit info
        remaining = self.rate_limiter.get_remaining_requests(user_id)
        
        status_message = f"""📊 **Your Mitra Status**

**Messages:** {user.total_messages}
**Account Created:** {user.created_at.strftime('%Y-%m-%d')}
**Last Active:** {user.last_active.strftime('%Y-%m-%d %H:%M')}

**Rate Limit:** {remaining}/{settings.rate_limit_messages_per_minute} messages available

Keep chatting! 💬"""
        
        await update.message.reply_text(status_message)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        correlation_id = str(uuid4())
        add_correlation_id(correlation_id)
        
        user = update.effective_user
        message_text = update.message.text
        user_id = str(user.id)
        
        logger.info(
            "message_received",
            user_id=user_id,
            message_length=len(message_text)
        )
        
        try:
            # Get or create user
            user_obj = self._get_or_create_user(user)
            user_obj.update_activity()
            
            # Check rate limit
            if not self.rate_limiter.is_allowed(user_id):
                remaining_time = self.rate_limiter.get_time_until_reset(user_id)
                await update.message.reply_text(
                    f"⏳ You're sending messages too quickly. Please wait {remaining_time} seconds before trying again."
                )
                return
            
            # Check message length
            if len(message_text) > settings.max_message_length:
                await update.message.reply_text(
                    f"📝 Your message is too long ({len(message_text)} characters). Please keep it under {settings.max_message_length} characters."
                )
                return
            
            # Send typing indicator
            await update.message.chat.send_action(action="typing")
            
            # Process message through engine
            response = await self.engine.process_message(
                user_id=user_id,
                message=message_text
            )
            
            # Send response
            await update.message.reply_text(response)
            
            logger.info(
                "message_handled_successfully",
                user_id=user_id,
                response_length=len(response)
            )
            
        except Exception as e:
            error = ErrorHandler.handle_error(e, user_id=user_id)
            await update.message.reply_text(error.user_facing_message)
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot."""
        logger.error(
            "telegram_error_occurred",
            error=str(context.error),
            update=str(update) if update else None
        )
        
        # Try to send error message to user
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "😓 I encountered an error processing your request. Please try again in a moment."
                )
            except TelegramError:
                pass
    
    def _get_or_create_user(self, telegram_user) -> User:
        """Get existing user or create a new one."""
        user_id = str(telegram_user.id)
        
        if user_id in self.users:
            return self.users[user_id]
        
        # Create new user
        user = User(
            id=user_id,
            username=telegram_user.username,
            first_name=telegram_user.first_name,
            last_name=telegram_user.last_name
        )
        
        self.users[user_id] = user
        
        logger.info(
            "user_created",
            user_id=user_id,
            username=telegram_user.username
        )
        
        return user
    
    def run(self) -> None:
        """Run the bot."""
        logger.info("starting_telegram_bot")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def start_webhook(self, webhook_url: str, port: int = 8443) -> None:
        """
        Start bot with webhook (for production deployment).
        
        Args:
            webhook_url: The webhook URL
            port: The port to listen on
        """
        logger.info("starting_telegram_bot_webhook", webhook_url=webhook_url, port=port)
        await self.app.bot.set_webhook(url=webhook_url)
        await self.app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path="webhook",
            webhook_url=webhook_url
        )
