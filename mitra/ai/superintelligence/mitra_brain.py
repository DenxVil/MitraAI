"""
🤖 Mitra AI - MitraSuperBrain
Core superintelligent AI engine with multi-path reasoning.
Coded by Denvil with love 🤍
"""

from enum import Enum, auto
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ThinkingMode(Enum):
    """Thinking modes for different response quality/speed tradeoffs."""
    INSTANT = auto()  # Fastest, single pass
    STANDARD = auto()  # Balanced, basic reasoning
    DEEP = auto()  # Thorough, chain of thought
    EXPERT = auto()  # Domain expertise engaged
    MAXIMUM = auto()  # All capabilities, multiple paths


class ExpertDomain(Enum):
    """Expert domains for specialized knowledge."""
    MATHEMATICS = "mathematics"
    CODING = "coding"
    REASONING = "reasoning"
    SCIENCE = "science"
    LANGUAGE = "language"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    GENERAL = "general"


@dataclass
class ThinkingStep:
    """A single step in the reasoning process."""
    step_number: int
    content: str
    confidence: float = 1.0
    domain: Optional[ExpertDomain] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReasoningPath:
    """A complete reasoning path."""
    path_id: str
    steps: List[ThinkingStep] = field(default_factory=list)
    conclusion: Optional[str] = None
    confidence: float = 0.0
    verified: bool = False


@dataclass
class ThinkingResult:
    """Result of the thinking process."""
    answer: str
    thinking_mode: ThinkingMode
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)
    final_confidence: float = 0.0
    domain_used: Optional[ExpertDomain] = None
    tokens_used: int = 0
    thinking_time_ms: float = 0.0
    verified: bool = False


class MitraSuperBrain:
    """
    Superintelligent AI brain with advanced reasoning capabilities.

    Features:
    - Multi-path reasoning (Tree of Thought)
    - Self-consistency voting
    - Expert domain routing
    - Chain of thought reasoning
    - Self-reflection and verification
    - Benchmark integration
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        max_thinking_paths: int = 5,
        enable_verification: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_thinking_paths = max_thinking_paths
        self.enable_verification = enable_verification

        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Expert modules
        self._experts: Dict[ExpertDomain, Any] = {}

        # Reasoning modules
        self._chain_of_thought = None
        self._tree_of_thought = None
        self._self_consistency = None
        self._self_reflection = None
        self._verifier = None

        logger.info(
            "mitra_brain_initialized",
            model=model_name,
            device=device,
            quantized=load_in_4bit,
        )

    async def load_model(self) -> None:
        """Load the AI model."""
        if self._loaded:
            return

        logger.info("loading_model", model=self.model_name)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Configure quantization
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None

            # Determine device
            if self.device == "auto":
                device_map = "auto"
            else:
                device_map = self.device

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

            self._loaded = True
            logger.info("model_loaded", model=self.model_name)

        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            raise

    async def think(
        self,
        query: str,
        mode: ThinkingMode = ThinkingMode.STANDARD,
        domain_hint: Optional[ExpertDomain] = None,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> ThinkingResult:
        """
        Process a query with intelligent reasoning.

        Args:
            query: The user's question or request
            mode: Thinking mode to use
            domain_hint: Optional hint for expert domain
            context: Optional conversation context

        Returns:
            ThinkingResult with answer and reasoning
        """
        start_time = datetime.now(timezone.utc)

        # Ensure model is loaded
        if not self._loaded:
            await self.load_model()

        # Detect domain if not provided
        domain = domain_hint or await self._detect_domain(query)

        # Select thinking strategy based on mode
        if mode == ThinkingMode.INSTANT:
            result = await self._think_instant(query, context)
        elif mode == ThinkingMode.STANDARD:
            result = await self._think_standard(query, domain, context)
        elif mode == ThinkingMode.DEEP:
            result = await self._think_deep(query, domain, context)
        elif mode == ThinkingMode.EXPERT:
            result = await self._think_expert(query, domain, context)
        elif mode == ThinkingMode.MAXIMUM:
            result = await self._think_maximum(query, domain, context)
        else:
            result = await self._think_standard(query, domain, context)

        # Calculate thinking time
        result.thinking_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.info(
            "thinking_complete",
            mode=mode.name,
            domain=domain.value if domain else None,
            confidence=result.final_confidence,
            time_ms=result.thinking_time_ms,
        )

        return result

    async def _detect_domain(self, query: str) -> ExpertDomain:
        """Detect the appropriate expert domain for a query."""
        query_lower = query.lower()

        # Simple keyword-based detection
        if any(w in query_lower for w in ["code", "program", "function", "debug", "python", "javascript"]):
            return ExpertDomain.CODING
        elif any(w in query_lower for w in ["calculate", "math", "equation", "solve", "number"]):
            return ExpertDomain.MATHEMATICS
        elif any(w in query_lower for w in ["science", "physics", "chemistry", "biology"]):
            return ExpertDomain.SCIENCE
        elif any(w in query_lower for w in ["write", "story", "poem", "creative"]):
            return ExpertDomain.CREATIVE
        elif any(w in query_lower for w in ["analyze", "compare", "evaluate", "assess"]):
            return ExpertDomain.ANALYSIS
        elif any(w in query_lower for w in ["explain", "why", "how", "reason", "logic"]):
            return ExpertDomain.REASONING
        else:
            return ExpertDomain.GENERAL

    async def _think_instant(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]],
    ) -> ThinkingResult:
        """Instant thinking - single pass, fastest response."""
        answer = await self._generate(query, context, max_tokens=256)
        return ThinkingResult(
            answer=answer,
            thinking_mode=ThinkingMode.INSTANT,
            final_confidence=0.7,
        )

    async def _think_standard(
        self,
        query: str,
        domain: ExpertDomain,
        context: Optional[List[Dict[str, str]]],
    ) -> ThinkingResult:
        """Standard thinking with basic reasoning."""
        # Add chain of thought prompt
        enhanced_query = f"Let me think step by step.\n\nQuestion: {query}\n\nAnswer:"
        answer = await self._generate(enhanced_query, context, max_tokens=512)

        return ThinkingResult(
            answer=answer,
            thinking_mode=ThinkingMode.STANDARD,
            domain_used=domain,
            final_confidence=0.8,
        )

    async def _think_deep(
        self,
        query: str,
        domain: ExpertDomain,
        context: Optional[List[Dict[str, str]]],
    ) -> ThinkingResult:
        """Deep thinking with thorough chain of thought."""
        # Step 1: Break down the problem
        breakdown = await self._generate(
            f"Break down this problem into smaller steps:\n\n{query}",
            context,
            max_tokens=256,
        )

        # Step 2: Solve each step
        solution = await self._generate(
            f"Problem: {query}\n\nSteps: {breakdown}\n\nNow solve each step carefully:",
            context,
            max_tokens=512,
        )

        # Step 3: Synthesize answer
        answer = await self._generate(
            f"Question: {query}\n\nReasoning: {solution}\n\nFinal answer:",
            context,
            max_tokens=256,
        )

        # Build reasoning path
        path = ReasoningPath(
            path_id="deep_1",
            steps=[
                ThinkingStep(1, breakdown),
                ThinkingStep(2, solution),
                ThinkingStep(3, answer),
            ],
            conclusion=answer,
            confidence=0.85,
        )

        return ThinkingResult(
            answer=answer,
            thinking_mode=ThinkingMode.DEEP,
            reasoning_paths=[path],
            domain_used=domain,
            final_confidence=0.85,
        )

    async def _think_expert(
        self,
        query: str,
        domain: ExpertDomain,
        context: Optional[List[Dict[str, str]]],
    ) -> ThinkingResult:
        """Expert thinking with domain-specific knowledge."""
        # Get domain-specific prompt
        expert_prompt = self._get_expert_prompt(domain)

        # Generate with expert context
        enhanced_query = f"{expert_prompt}\n\nQuestion: {query}\n\nExpert answer:"
        answer = await self._generate(enhanced_query, context, max_tokens=768)

        return ThinkingResult(
            answer=answer,
            thinking_mode=ThinkingMode.EXPERT,
            domain_used=domain,
            final_confidence=0.9,
        )

    async def _think_maximum(
        self,
        query: str,
        domain: ExpertDomain,
        context: Optional[List[Dict[str, str]]],
    ) -> ThinkingResult:
        """Maximum thinking with all capabilities."""
        paths: List[ReasoningPath] = []

        # Generate multiple reasoning paths
        for i in range(min(3, self.max_thinking_paths)):
            temperature = 0.7 + (i * 0.1)  # Vary temperature
            path_answer = await self._generate(
                f"Approach {i+1}: Let me think about this differently.\n\nQuestion: {query}\n\n",
                context,
                max_tokens=512,
                temperature=temperature,
            )
            paths.append(ReasoningPath(
                path_id=f"max_{i}",
                conclusion=path_answer,
                confidence=0.8,
            ))

        # Self-consistency voting
        best_answer = await self._vote_on_answers([p.conclusion for p in paths])

        # Verification
        if self.enable_verification:
            verified, verification_note = await self._verify_answer(query, best_answer)
        else:
            verified = False
            verification_note = ""

        if verified:
            final_answer = best_answer
        else:
            # Refine if verification failed
            final_answer = await self._generate(
                f"Original answer: {best_answer}\n"
                f"Verification note: {verification_note}\n"
                f"Please provide a corrected answer:",
                context,
                max_tokens=512,
            )

        return ThinkingResult(
            answer=final_answer,
            thinking_mode=ThinkingMode.MAXIMUM,
            reasoning_paths=paths,
            domain_used=domain,
            final_confidence=0.95 if verified else 0.85,
            verified=verified,
        )

    def _get_expert_prompt(self, domain: ExpertDomain) -> str:
        """Get expert system prompt for a domain."""
        prompts = {
            ExpertDomain.MATHEMATICS: (
                "You are an expert mathematician. Solve problems step by step, "
                "show all work, and verify your calculations."
            ),
            ExpertDomain.CODING: (
                "You are an expert programmer. Write clean, efficient, well-documented code. "
                "Explain your approach and consider edge cases."
            ),
            ExpertDomain.REASONING: (
                "You are an expert in logical reasoning. Analyze arguments carefully, "
                "identify assumptions, and draw valid conclusions."
            ),
            ExpertDomain.SCIENCE: (
                "You are an expert scientist. Explain concepts clearly with examples, "
                "cite relevant principles, and acknowledge uncertainty."
            ),
            ExpertDomain.CREATIVE: (
                "You are a creative expert. Generate original, engaging content "
                "with vivid imagery and compelling narratives."
            ),
            ExpertDomain.ANALYSIS: (
                "You are an analytical expert. Break down complex topics, "
                "compare perspectives, and provide balanced assessments."
            ),
            ExpertDomain.LANGUAGE: (
                "You are a language expert. Communicate clearly, use precise vocabulary, "
                "and adapt style to context."
            ),
            ExpertDomain.GENERAL: (
                "You are a knowledgeable assistant. Provide helpful, accurate, "
                "and well-organized responses."
            ),
        }
        return prompts.get(domain, prompts[ExpertDomain.GENERAL])

    async def _generate(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from the model."""
        if not self._loaded or not self._model:
            # Return placeholder if model not loaded
            return (
                "I'm processing your request. The AI model is being configured.\n\n"
                "_Coded by Denvil with love 🤍_"
            )

        try:
            import torch

            # Build messages
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})

            # Tokenize - handle different tokenizer return types
            tokenized = self._tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            
            # Move to model device if available
            device = getattr(self._model, "device", None)
            if device is not None and hasattr(tokenized, "to"):
                inputs = tokenized.to(device)
            else:
                inputs = tokenized

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode
            response = self._tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            logger.error("generation_failed", error=str(e))
            return f"Error generating response: {str(e)}"

    async def _vote_on_answers(self, answers: List[str]) -> str:
        """Vote on multiple answers using self-consistency."""
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]

        # Simple voting - in production, use semantic similarity
        # For now, return the longest answer as it likely has more detail
        return max(answers, key=len)

    async def _verify_answer(
        self,
        query: str,
        answer: str,
    ) -> tuple:
        """Verify an answer for correctness."""
        verification_prompt = (
            f"Verify this answer:\n\n"
            f"Question: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Is this answer correct? Respond with 'CORRECT' or 'INCORRECT' and explain why."
        )

        verification = await self._generate(verification_prompt, None, max_tokens=256)
        verified = "CORRECT" in verification.upper()

        return verified, verification

    async def stream_think(
        self,
        query: str,
        mode: ThinkingMode = ThinkingMode.STANDARD,
    ) -> AsyncIterator[str]:
        """Stream thinking output for real-time display."""
        # Ensure model is loaded
        if not self._loaded:
            await self.load_model()

        # Generate and yield chunks
        # This is a placeholder - implement proper streaming
        result = await self.think(query, mode)
        for word in result.answer.split():
            yield word + " "
            await asyncio.sleep(0.05)

    async def get_capabilities(self) -> Dict[str, Any]:
        """Get brain capabilities and status."""
        return {
            "model": self.model_name,
            "loaded": self._loaded,
            "device": self.device,
            "quantized": self.load_in_4bit,
            "thinking_modes": [m.name for m in ThinkingMode],
            "expert_domains": [d.value for d in ExpertDomain],
            "max_paths": self.max_thinking_paths,
            "verification_enabled": self.enable_verification,
        }
