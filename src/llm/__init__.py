"""
LLM Integration module для Semantic Contamination

Предоставляет унифицированный интерфейс для работы с различными LLM провайдерами:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)

Пример использования:
    from llm import get_default_llm, OpenAIClient

    # Использование default LLM из config
    llm = get_default_llm()
    response = llm.generate("Your prompt here")

    # Или напрямую создать клиент
    llm = OpenAIClient(model="gpt-4-turbo-preview")
    response = llm.generate("Your prompt here")
"""

from .base import LLMClient, LLMResponse
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .cost_tracker import CostTracker
from .utils import get_default_llm, count_tokens

__all__ = [
    "LLMClient",
    "LLMResponse",
    "OpenAIClient",
    "AnthropicClient",
    "CostTracker",
    "get_default_llm",
    "count_tokens",
]
