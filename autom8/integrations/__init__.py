"""
External Integrations

Integrations with external services like Ollama, cloud providers, and other tools.
"""

from autom8.integrations.ollama import OllamaClient
from autom8.integrations.anthropic import AnthropicClient, get_anthropic_client
from autom8.integrations.openai import OpenAIClient, get_openai_client

__all__ = [
    "OllamaClient",
    "AnthropicClient", 
    "get_anthropic_client",
    "OpenAIClient",
    "get_openai_client",
]
