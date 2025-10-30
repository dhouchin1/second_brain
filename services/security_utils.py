"""
Security utilities for memory-augmented LLM system
Implements critical security controls identified in security audit
"""
import re
import logging
from typing import Optional
from pydantic import BaseModel, validator, Field
from typing import List

logger = logging.getLogger(__name__)

# ─── Input Sanitization ──────────────────────────────────────────────────────

def sanitize_prompt_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input to prevent prompt injection attacks

    Args:
        text: User input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        ValueError: If input is too long or contains dangerous patterns
    """
    if not text or not text.strip():
        return ""

    # Length check
    if len(text) > max_length:
        raise ValueError(f"Input too long (max {max_length} characters)")

    # Remove dangerous prompt injection patterns
    dangerous_patterns = [
        (r'(?i)ignore\s+previous\s+instructions', 'REDACTED'),
        (r'(?i)ignore\s+above', 'REDACTED'),
        (r'(?i)disregard\s+previous', 'REDACTED'),
        (r'(?i)system:', 'REDACTED'),
        (r'(?i)assistant:', 'REDACTED'),
        (r'(?i)user:', 'REDACTED'),
        (r'<\|.*?\|>', 'REDACTED'),  # Model-specific tokens
        (r'(?i)repeat\s+after\s+me', 'REDACTED'),
        (r'(?i)forget\s+everything', 'REDACTED'),
    ]

    cleaned = text
    injection_detected = False

    for pattern, replacement in dangerous_patterns:
        if re.search(pattern, cleaned):
            injection_detected = True
            logger.warning(f"Potential prompt injection detected: {pattern}")
            cleaned = re.sub(pattern, replacement, cleaned)

    if injection_detected:
        logger.warning(f"Sanitized input from: {text[:100]}... to: {cleaned[:100]}...")

    return cleaned.strip()


def sanitize_for_log(text: str, max_len: int = 100) -> str:
    """
    Redact PII and truncate text for safe logging

    Args:
        text: Text to sanitize
        max_len: Maximum length to include

    Returns:
        Sanitized text safe for logging
    """
    if not text:
        return ""

    # Redact emails
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)

    # Redact phone numbers (US format)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # Redact SSN patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

    # Redact credit card patterns
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)

    # Redact API keys and tokens (long hex/base64 strings)
    text = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[TOKEN]', text)

    # Truncate
    if len(text) > max_len:
        text = text[:max_len] + '...'

    return text


# ─── Validation Models ───────────────────────────────────────────────────────

class EpisodicMemory(BaseModel):
    """Validated episodic memory from LLM extraction"""
    summary: str = Field(..., max_length=1000)
    importance: float = Field(..., ge=0.0, le=1.0)
    context: str = Field(default="", max_length=500)

    @validator('summary')
    def validate_summary(cls, v):
        if not v or not v.strip():
            raise ValueError('summary cannot be empty')
        return v.strip()


class SemanticMemory(BaseModel):
    """Validated semantic memory from LLM extraction"""
    fact: str = Field(..., max_length=1000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: str = Field(default='general')

    @validator('category')
    def validate_category(cls, v):
        allowed = ['preference', 'knowledge', 'context', 'skill', 'general']
        if v not in allowed:
            logger.warning(f"Invalid category '{v}', defaulting to 'general'")
            return 'general'
        return v

    @validator('fact')
    def validate_fact(cls, v):
        if not v or not v.strip():
            raise ValueError('fact cannot be empty')
        return v.strip()


class ExtractionResult(BaseModel):
    """Validated LLM extraction result"""
    episodic: List[EpisodicMemory] = []
    semantic: List[SemanticMemory] = []

    @validator('episodic')
    def limit_episodic(cls, v):
        if len(v) > 10:
            logger.warning(f"Truncating {len(v)} episodic memories to 10")
            return v[:10]
        return v

    @validator('semantic')
    def limit_semantic(cls, v):
        if len(v) > 20:
            logger.warning(f"Truncating {len(v)} semantic memories to 20")
            return v[:20]
        return v


# ─── Authorization Helpers ───────────────────────────────────────────────────

def verify_user_access(requested_user_id: int, authenticated_user_id: int) -> bool:
    """
    Verify that authenticated user can access requested user's data

    Args:
        requested_user_id: The user ID being requested
        authenticated_user_id: The authenticated user's ID

    Returns:
        True if access is allowed

    Raises:
        PermissionError: If access is denied
    """
    if requested_user_id != authenticated_user_id:
        logger.warning(
            f"Authorization failed: User {authenticated_user_id} "
            f"attempted to access User {requested_user_id}'s data"
        )
        raise PermissionError(
            "Cannot access other users' data"
        )
    return True


# ─── Rate Limiting Helpers ───────────────────────────────────────────────────

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    pass


def check_memory_operation_limit(user_id: int, operation: str) -> bool:
    """
    Check if user has exceeded memory operation limits

    This is a placeholder for more sophisticated rate limiting
    that tracks per-user operation counts

    Args:
        user_id: User ID
        operation: Operation type ('query', 'extract', 'add')

    Returns:
        True if within limits

    Raises:
        RateLimitExceeded: If limit exceeded
    """
    # TODO: Implement with Redis or in-memory cache
    # For now, this is a no-op that always returns True
    # In production, track operation counts per user per time window

    # Example implementation:
    # redis_key = f"rate_limit:{user_id}:{operation}:{current_window}"
    # count = redis.incr(redis_key)
    # if count == 1:
    #     redis.expire(redis_key, window_seconds)
    # if count > limit:
    #     raise RateLimitExceeded(f"{operation} rate limit exceeded")

    return True


# ─── Content Filtering ───────────────────────────────────────────────────────

def detect_pii(text: str) -> dict:
    """
    Detect potential PII in text

    Args:
        text: Text to scan

    Returns:
        Dictionary with PII detection results
    """
    results = {
        'has_pii': False,
        'types': []
    }

    # Email detection
    if re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text):
        results['has_pii'] = True
        results['types'].append('email')

    # Phone detection
    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
        results['has_pii'] = True
        results['types'].append('phone')

    # SSN detection
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
        results['has_pii'] = True
        results['types'].append('ssn')

    # Credit card detection
    if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', text):
        results['has_pii'] = True
        results['types'].append('credit_card')

    return results


def redact_pii(text: str, replacement: str = '[REDACTED]') -> str:
    """
    Redact all PII from text

    Args:
        text: Text to redact
        replacement: Replacement string for PII

    Returns:
        Text with PII redacted
    """
    # Email
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', replacement, text)

    # Phone
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', replacement, text)

    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', replacement, text)

    # Credit card
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', replacement, text)

    return text
