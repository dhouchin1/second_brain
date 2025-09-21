"""
Token Counting Utilities

Accurate token counting for different models and providers,
supporting both local and cloud model tokenization.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Union

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class TokenizerType(str, Enum):
    """Types of tokenizers available"""
    TIKTOKEN = "tiktoken"      # OpenAI's tiktoken
    HUGGINGFACE = "huggingface" # HuggingFace tokenizers
    ESTIMATION = "estimation"   # Character-based estimation
    CLAUDE = "claude"          # Anthropic's tokenization


class TokenCounter:
    """
    Multi-model token counting with automatic model detection
    """
    
    def __init__(self):
        self.encoders = {}
        self._initialize_encoders()
    
    def _initialize_encoders(self):
        """Initialize available tokenizer encoders"""
        if TIKTOKEN_AVAILABLE:
            try:
                # Common OpenAI models
                self.encoders["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
                self.encoders["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                self.encoders["text-davinci-003"] = tiktoken.encoding_for_model("text-davinci-003")
                
                # Common encodings
                self.encoders["cl100k_base"] = tiktoken.get_encoding("cl100k_base")  # GPT-4, GPT-3.5
                self.encoders["p50k_base"] = tiktoken.get_encoding("p50k_base")      # Codex models
                
                logger.debug("Initialized tiktoken encoders")
            except Exception as e:
                logger.warning(f"Failed to initialize some tiktoken encoders: {e}")
    
    def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> int:
        """
        Count tokens for given text and model
        
        Args:
            text: Text to count tokens for
            model: Model name (e.g., 'gpt-4', 'claude-sonnet')
            encoding: Specific encoding to use
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        # Determine the best tokenization method
        tokenizer_type = self._get_tokenizer_type(model, encoding)
        
        if tokenizer_type == TokenizerType.TIKTOKEN:
            return self._count_tiktoken(text, model, encoding)
        elif tokenizer_type == TokenizerType.CLAUDE:
            return self._count_claude_tokens(text)
        elif tokenizer_type == TokenizerType.HUGGINGFACE:
            return self._count_huggingface_tokens(text, model)
        else:
            return self._estimate_tokens(text)
    
    def _get_tokenizer_type(self, model: Optional[str], encoding: Optional[str]) -> TokenizerType:
        """Determine the best tokenizer for the given model"""
        if encoding and encoding in self.encoders:
            return TokenizerType.TIKTOKEN
        
        if model:
            # OpenAI models
            if any(name in model.lower() for name in ["gpt", "davinci", "curie", "babbage", "ada"]):
                return TokenizerType.TIKTOKEN
            
            # Anthropic models
            if any(name in model.lower() for name in ["claude", "sonnet", "haiku", "opus"]):
                return TokenizerType.CLAUDE
            
            # Local models - try to detect type
            if any(name in model.lower() for name in ["llama", "mixtral", "mistral", "phi"]):
                return TokenizerType.HUGGINGFACE
        
        # Default to estimation
        return TokenizerType.ESTIMATION
    
    def _count_tiktoken(self, text: str, model: Optional[str], encoding: Optional[str]) -> int:
        """Count tokens using tiktoken"""
        if not TIKTOKEN_AVAILABLE:
            logger.warning("tiktoken not available, falling back to estimation")
            return self._estimate_tokens(text)
        
        try:
            # Try specific model first
            if model and model in self.encoders:
                encoder = self.encoders[model]
            # Then try specific encoding
            elif encoding and encoding in self.encoders:
                encoder = self.encoders[encoding]
            # Default to GPT-4 encoding
            else:
                encoder = self.encoders.get("cl100k_base")
            
            if encoder:
                return len(encoder.encode(text))
            else:
                logger.warning(f"No encoder found for model {model}, using estimation")
                return self._estimate_tokens(text)
                
        except Exception as e:
            logger.warning(f"tiktoken encoding failed: {e}, using estimation")
            return self._estimate_tokens(text)
    
    def _count_claude_tokens(self, text: str) -> int:
        """
        Count tokens for Claude models using estimation
        
        Claude uses a different tokenization that's not publicly available,
        so we use a calibrated estimation based on observed behavior.
        """
        # Claude's tokenization is roughly similar to GPT but with some differences
        # Based on testing, Claude tokens are approximately:
        # - English: ~4 characters per token
        # - Code: ~3.5 characters per token
        # - Mixed content: ~3.8 characters per token
        
        # Detect content type
        code_patterns = [
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',         # Inline code
            r'def\s+\w+\(',     # Function definitions
            r'class\s+\w+\(',   # Class definitions
            r'import\s+\w+',    # Imports
            r'from\s+\w+\s+import',  # From imports
        ]
        
        code_content = sum(len(match.group()) for pattern in code_patterns 
                          for match in re.finditer(pattern, text))
        
        total_chars = len(text)
        code_ratio = code_content / total_chars if total_chars > 0 else 0
        
        # Adjust estimation based on content type
        if code_ratio > 0.5:
            # Mostly code
            chars_per_token = 3.5
        elif code_ratio > 0.2:
            # Mixed content
            chars_per_token = 3.8
        else:
            # Mostly text
            chars_per_token = 4.0
        
        estimated_tokens = total_chars / chars_per_token
        
        # Add small buffer for special tokens
        return int(estimated_tokens * 1.05)
    
    def _count_huggingface_tokens(self, text: str, model: Optional[str]) -> int:
        """
        Count tokens for HuggingFace models
        
        For local models, we don't have the exact tokenizer available,
        so we use educated estimation based on model family.
        """
        # Different model families have different tokenization characteristics
        if model:
            model_lower = model.lower()
            
            # Llama family (includes Code Llama)
            if "llama" in model_lower:
                # Llama uses SentencePiece, roughly 3.5-4 chars per token
                chars_per_token = 3.7
            
            # Mixtral/Mistral family
            elif any(name in model_lower for name in ["mixtral", "mistral"]):
                # Similar to Llama tokenization
                chars_per_token = 3.8
            
            # Phi family
            elif "phi" in model_lower:
                # CodeGen-style tokenization
                chars_per_token = 3.9
            
            else:
                # Default estimation
                chars_per_token = 3.8
        else:
            chars_per_token = 3.8
        
        # Estimate tokens
        estimated_tokens = len(text) / chars_per_token
        
        # Add buffer for special tokens and subword tokenization variance
        return int(estimated_tokens * 1.1)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Fallback token estimation using character-based heuristics
        
        This provides a reasonable estimate when specific tokenizers
        are not available.
        """
        if not text:
            return 0
        
        # Base estimation: ~4 characters per token (GPT-like)
        base_estimate = len(text) / 4.0
        
        # Adjust for different content types
        
        # Count whitespace - tokens often align with word boundaries
        words = len(text.split())
        whitespace_factor = words / len(text) if len(text) > 0 else 0
        
        # Count punctuation - often becomes separate tokens
        punctuation = len(re.findall(r'[^\w\s]', text))
        punctuation_factor = punctuation / len(text) if len(text) > 0 else 0
        
        # Adjust estimation
        adjusted_estimate = base_estimate
        
        # More words = slightly more tokens
        adjusted_estimate *= (1 + whitespace_factor * 0.1)
        
        # More punctuation = slightly more tokens
        adjusted_estimate *= (1 + punctuation_factor * 0.2)
        
        return max(1, int(adjusted_estimate))
    
    def estimate_cost(
        self,
        text: str,
        model: str,
        is_input: bool = True,
        cost_per_token: Optional[float] = None
    ) -> float:
        """
        Estimate cost for processing text with given model
        
        Args:
            text: Text to process
            model: Model name
            is_input: True for input tokens, False for output tokens
            cost_per_token: Override cost per token
            
        Returns:
            Estimated cost in USD
        """
        if cost_per_token is None:
            cost_per_token = self._get_model_cost_per_token(model, is_input)
        
        if cost_per_token == 0.0:
            return 0.0  # Local model
        
        token_count = self.count_tokens(text, model)
        return token_count * cost_per_token
    
    def _get_model_cost_per_token(self, model: str, is_input: bool) -> float:
        """Get cost per token for a model (hardcoded for common models)"""
        
        # Cost data (as of 2024) - should be configurable in production
        costs = {
            # OpenAI models (input, output)
            "gpt-4": (0.03, 0.06),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0015, 0.002),
            
            # Anthropic models
            "claude-opus": (0.015, 0.075),
            "claude-sonnet": (0.003, 0.015),
            "claude-haiku": (0.00025, 0.00125),
            
            # Local models (free)
            "llama": (0.0, 0.0),
            "mixtral": (0.0, 0.0),
            "mistral": (0.0, 0.0),
            "phi": (0.0, 0.0),
        }
        
        # Find matching model
        model_lower = model.lower()
        for model_key, (input_cost, output_cost) in costs.items():
            if model_key in model_lower:
                return input_cost if is_input else output_cost
        
        # Default to free (local model)
        return 0.0


# Global token counter instance
_token_counter = None


def get_token_counter() -> TokenCounter:
    """Get global token counter instance"""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def estimate_tokens(
    text: str,
    model: Optional[str] = None,
    encoding: Optional[str] = None
) -> int:
    """
    Convenience function to estimate tokens
    
    Args:
        text: Text to count tokens for
        model: Model name for better accuracy
        encoding: Specific encoding to use
        
    Returns:
        Estimated token count
    """
    return get_token_counter().count_tokens(text, model, encoding)


def estimate_cost(
    text: str,
    model: str,
    is_input: bool = True
) -> float:
    """
    Convenience function to estimate cost
    
    Args:
        text: Text to process
        model: Model name
        is_input: True for input tokens, False for output tokens
        
    Returns:
        Estimated cost in USD
    """
    return get_token_counter().estimate_cost(text, model, is_input)


def count_tokens_in_messages(messages: List[Dict[str, str]], model: Optional[str] = None) -> int:
    """
    Count tokens in a list of chat messages
    
    Args:
        messages: List of messages with 'role' and 'content' keys
        model: Model name for accurate counting
        
    Returns:
        Total token count including message formatting overhead
    """
    counter = get_token_counter()
    total_tokens = 0
    
    for message in messages:
        # Count content tokens
        content = message.get('content', '')
        content_tokens = counter.count_tokens(content, model)
        
        # Add overhead for message formatting
        # Different models have different overhead, but typically:
        # - Role tokens (system/user/assistant): ~1-2 tokens
        # - Message separator tokens: ~2-4 tokens
        role = message.get('role', 'user')
        if role == 'system':
            overhead = 4  # System messages have slightly more overhead
        else:
            overhead = 3  # User/assistant messages
        
        total_tokens += content_tokens + overhead
    
    # Add conversation-level overhead (varies by model)
    if model and any(name in model.lower() for name in ["gpt", "claude"]):
        conversation_overhead = 3  # Most chat models add a few tokens
    else:
        conversation_overhead = 2
    
    return total_tokens + conversation_overhead


def optimize_for_token_budget(
    text: str,
    budget: int,
    model: Optional[str] = None,
    preserve_start: int = 100,
    preserve_end: int = 100
) -> str:
    """
    Truncate text to fit within token budget while preserving important parts
    
    Args:
        text: Original text
        budget: Maximum tokens allowed
        model: Model for accurate token counting
        preserve_start: Characters to preserve from start
        preserve_end: Characters to preserve from end
        
    Returns:
        Truncated text that fits within budget
    """
    counter = get_token_counter()
    current_tokens = counter.count_tokens(text, model)
    
    if current_tokens <= budget:
        return text
    
    # Calculate how much we need to remove
    chars_per_token = len(text) / current_tokens
    target_chars = int(budget * chars_per_token * 0.9)  # 10% buffer
    
    if target_chars <= preserve_start + preserve_end:
        # Can't preserve requested amounts, just truncate from end
        target_chars = max(100, target_chars)
        truncated = text[:target_chars] + "..."
    else:
        # Preserve start and end, truncate middle
        start_part = text[:preserve_start]
        end_part = text[-preserve_end:]
        middle_budget = target_chars - preserve_start - preserve_end - 20  # Account for ellipsis
        
        if middle_budget > 0:
            middle_start = preserve_start
            middle_end = len(text) - preserve_end
            middle_chars = min(middle_budget, middle_end - middle_start)
            middle_part = text[middle_start:middle_start + middle_chars]
            truncated = start_part + middle_part + "...[truncated]..." + end_part
        else:
            truncated = start_part + "...[truncated]..." + end_part
    
    # Verify it fits
    final_tokens = counter.count_tokens(truncated, model)
    if final_tokens > budget:
        # Still too long, do simple truncation
        chars_per_token = len(truncated) / final_tokens
        target_chars = int(budget * chars_per_token * 0.8)
        truncated = text[:target_chars] + "..."
    
    return truncated