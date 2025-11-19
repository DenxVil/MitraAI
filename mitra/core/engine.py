"""
Core AI engine for Mitra.

Orchestrates AI responses with emotion awareness, multi-step reasoning,
and safety features.
"""

from typing import Optional
import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..models import Conversation, Message, MessageRole, EmotionalContext
from ..utils import get_logger, ErrorHandler, ErrorCategory, MitraError
from .emotion_analyzer import EmotionAnalyzer
from .safety_filter import SafetyFilter
from .prompts import PromptBuilder, SystemPrompts


logger = get_logger(__name__)


class MitraEngine:
    """
    Core intelligence engine for Mitra AI.

    Handles conversation management, emotion analysis, reasoning,
    and response generation with safety features.
    """

    def __init__(self):
        """Initialize the Mitra engine."""
        # Initialize AI client
        if settings.use_azure_openai:
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
            )
            self.model = settings.azure_openai_deployment_name
        else:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            self.model = "gpt-4"

        # Initialize components
        self.emotion_analyzer = EmotionAnalyzer()
        self.safety_filter = SafetyFilter(enable_moderation=settings.enable_content_moderation)
        self.prompt_builder = PromptBuilder()

        # Conversation storage (in-memory for now)
        # In production, use a database
        self.conversations: dict[str, Conversation] = {}

        logger.info(
            "mitra_engine_initialized",
            model=self.model,
            use_azure=settings.use_azure_openai,
            moderation_enabled=settings.enable_content_moderation,
        )

    async def process_message(
        self, user_id: str, message: str, conversation_id: Optional[str] = None
    ) -> str:
        """
        Process a user message and generate a response.

        Args:
            user_id: The user's identifier
            message: The user's message text
            conversation_id: Optional conversation ID to continue existing conversation

        Returns:
            Mitra's response text
        """
        logger.info(
            "processing_message",
            user_id=user_id,
            conversation_id=conversation_id,
            message_length=len(message),
        )

        try:
            # Get or create conversation
            conversation = self._get_or_create_conversation(user_id, conversation_id)

            # Check message safety
            should_refuse, refusal_reason = self.safety_filter.should_refuse(message)

            if should_refuse:
                response = self.safety_filter.get_boundary_response(refusal_reason)
                conversation.add_message(MessageRole.USER, message)
                conversation.add_message(MessageRole.ASSISTANT, response)
                return response

            # Check for crisis situation
            if refusal_reason == "crisis":
                response = self.safety_filter.get_crisis_response()
                conversation.add_message(MessageRole.USER, message)
                conversation.add_message(MessageRole.ASSISTANT, response)
                return response

            # Analyze emotional content
            emotional_context = self.emotion_analyzer.analyze(message)

            # Add user message to conversation
            conversation.add_message(MessageRole.USER, message, emotional_context=emotional_context)

            # Generate response
            response = await self._generate_response(conversation, emotional_context)

            # Add assistant response to conversation
            conversation.add_message(MessageRole.ASSISTANT, response)

            logger.info(
                "message_processed_successfully",
                user_id=user_id,
                conversation_id=conversation.id,
                response_length=len(response),
            )

            return response

        except Exception as e:
            error = ErrorHandler.handle_error(e, user_id=user_id)
            return error.user_facing_message

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _generate_response(
        self, conversation: Conversation, emotional_context: EmotionalContext
    ) -> str:
        """
        Generate AI response with retry logic.

        Args:
            conversation: The conversation context
            emotional_context: Detected emotional context

        Returns:
            Generated response text
        """
        try:
            # Build messages for API
            messages = self._build_api_messages(conversation, emotional_context)

            # Call AI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3,
            )

            response_text = response.choices[0].message.content.strip()

            logger.debug(
                "ai_response_generated",
                conversation_id=conversation.id,
                tokens_used=response.usage.total_tokens if response.usage else None,
            )

            return response_text

        except Exception as e:
            logger.error("ai_generation_failed", error=str(e), conversation_id=conversation.id)
            raise MitraError(
                message=f"AI generation failed: {str(e)}",
                category=ErrorCategory.AI_SERVICE,
                original_error=e,
            )

    def _build_api_messages(
        self, conversation: Conversation, emotional_context: EmotionalContext
    ) -> list[dict]:
        """Build message list for AI API."""
        messages = []

        # Add system prompt
        system_prompt = self.prompt_builder.build_system_prompt()

        # Enhance system prompt based on emotional context
        if emotional_context.needs_support:
            system_prompt += "\n\nThe user appears to need emotional support. Please be extra empathetic, validating, and supportive in your response."

        messages.append({"role": "system", "content": system_prompt})

        # Add conversation history (limited)
        history = conversation.get_recent_messages(settings.max_conversation_history)
        for msg in history:
            messages.append({"role": msg.role.value, "content": msg.content})

        return messages

    def _get_or_create_conversation(
        self, user_id: str, conversation_id: Optional[str] = None
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]

        # Create new conversation
        conversation = Conversation(user_id=user_id)
        self.conversations[conversation.id] = conversation

        logger.debug("conversation_created", user_id=user_id, conversation_id=conversation.id)

        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear a conversation from memory.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if conversation was found and cleared
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info("conversation_cleared", conversation_id=conversation_id)
            return True
        return False
