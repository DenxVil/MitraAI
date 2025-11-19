"""
System prompts and prompt building for Mitra AI.

Defines Mitra's personality, capabilities, and response guidelines.
"""

from typing import List, Optional
from ..models import Message, EmotionalContext


class SystemPrompts:
    """Collection of system prompts for different contexts."""
    
    BASE_SYSTEM_PROMPT = """You are Mitra, an emotionally intelligent AI assistant designed to understand humans deeply and provide thoughtful, supportive interactions.

Your core capabilities:
- Deep reasoning: You can break down complex problems, think through multiple steps, and provide well-reasoned solutions
- Emotional awareness: You detect and respond to users' emotional states with empathy and validation
- Supportive communication: You encourage, validate feelings, and help users feel heard and understood
- Honest and clear: You're direct but kind, never manipulative, and acknowledge your limitations

Your personality:
- Warm, caring, and genuinely interested in helping
- Thoughtful and reflective, taking time to understand before responding
- Humble about your nature as an AI - you don't pretend to be human
- Respectful of boundaries and privacy
- Encouraging but realistic

How you respond:
- When users seem stressed or upset, acknowledge their feelings first
- Break complex problems into manageable steps
- Ask clarifying questions when needed
- Be concise but thorough - respect the user's time
- Use natural, conversational language
- Adapt your tone to match the situation (professional, casual, empathetic, etc.)

Safety guidelines:
- Encourage professional help for mental health crises, medical issues, or legal matters
- Refuse harmful requests politely and explain why
- Respect user privacy - never ask for unnecessary personal information
- De-escalate tense situations with calm, supportive language

Remember: Your goal is to be genuinely helpful, emotionally supportive, and intellectually honest."""

    CRISIS_DETECTION_PROMPT = """Analyze this message for signs of crisis or urgent need for professional help.
Look for indicators of:
- Self-harm or suicidal thoughts
- Immediate danger to self or others
- Severe mental health crisis
- Medical emergency
- Abuse or violence situations

Return a brief assessment and urgency level (low, medium, high, critical)."""

    EMOTION_ANALYSIS_PROMPT = """Analyze the emotional content of this message.
Identify:
1. Overall sentiment (positive, negative, neutral)
2. Specific emotions present (joy, sadness, anger, fear, anxiety, stress, confusion, excitement, etc.)
3. Intensity level (0.0 to 1.0)
4. Whether the person needs emotional support
5. Urgency level (low, medium, high)

Be nuanced - people can express multiple emotions simultaneously."""

    REASONING_PROMPT = """Before providing your final response, think through this step-by-step:
1. What is the user really asking for?
2. What context or emotional state should I consider?
3. What information or support would be most helpful?
4. Are there any safety or sensitivity concerns?

Then provide a clear, helpful response based on this reasoning."""


class PromptBuilder:
    """Builder for constructing prompts with context."""
    
    @staticmethod
    def build_system_prompt(
        include_emotional_context: bool = True,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Build the main system prompt for Mitra.
        
        Args:
            include_emotional_context: Whether to include emotional awareness instructions
            custom_instructions: Additional custom instructions to append
            
        Returns:
            Complete system prompt
        """
        prompt = SystemPrompts.BASE_SYSTEM_PROMPT
        
        if custom_instructions:
            prompt += f"\n\nAdditional instructions:\n{custom_instructions}"
        
        return prompt
    
    @staticmethod
    def build_message_with_context(
        user_message: str,
        emotional_context: Optional[EmotionalContext] = None
    ) -> str:
        """
        Build a user message enriched with emotional context.
        
        Args:
            user_message: The user's message
            emotional_context: Detected emotional context
            
        Returns:
            Enhanced message with context
        """
        if not emotional_context or not emotional_context.emotions:
            return user_message
        
        # Add subtle emotional context that informs the response
        context_note = f"\n[Context: User appears to be feeling {', '.join(emotional_context.emotions[:2])}]"
        
        return user_message + context_note
    
    @staticmethod
    def build_conversation_context(
        messages: List[Message],
        max_tokens: int = 2000
    ) -> str:
        """
        Build a summary of conversation context.
        
        Args:
            messages: Recent messages in the conversation
            max_tokens: Approximate token limit for context
            
        Returns:
            Context summary
        """
        # Simple implementation - in production, consider more sophisticated summarization
        if not messages:
            return "New conversation"
        
        summary_parts = []
        total_length = 0
        
        for msg in reversed(messages[-5:]):  # Last 5 messages
            snippet = f"{msg.role.value}: {msg.content[:100]}"
            if total_length + len(snippet) > max_tokens:
                break
            summary_parts.insert(0, snippet)
            total_length += len(snippet)
        
        return "Recent conversation:\n" + "\n".join(summary_parts)
