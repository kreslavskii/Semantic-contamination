"""
Utility функции для LLM интеграции
"""
import logging
from typing import Optional
import tiktoken

from .base import LLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from ..config import settings

logger = logging.getLogger(__name__)


def get_default_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> LLMClient:
    """
    Получить default LLM клиент на основе конфигурации

    Args:
        model: Название модели (если None, используется из settings)
        temperature: Температура (если None, используется из settings)
        max_tokens: Максимум токенов (если None, используется из settings)
        **kwargs: Дополнительные параметры для клиента

    Returns:
        LLMClient (OpenAI или Anthropic)

    Raises:
        ValueError: Если API ключи не настроены

    Пример:
        llm = get_default_llm()
        response = llm.generate("What is the meaning of life?")
    """
    provider = settings.DEFAULT_LLM_PROVIDER
    model = model or settings.DEFAULT_MODEL
    temperature = temperature if temperature is not None else settings.TEMPERATURE
    max_tokens = max_tokens if max_tokens is not None else settings.MAX_TOKENS

    if provider == "openai":
        if not settings.has_openai_key:
            raise ValueError(
                "OpenAI API key не найден. Установите OPENAI_API_KEY в .env"
            )
        return OpenAIClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    elif provider == "anthropic":
        if not settings.has_anthropic_key:
            raise ValueError(
                "Anthropic API key не найден. Установите ANTHROPIC_API_KEY в .env"
            )
        return AnthropicClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    else:
        raise ValueError(f"Неизвестный провайдер: {provider}")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Подсчет токенов в тексте

    Args:
        text: Текст для подсчета
        model: Модель для tokenizer (по умолчанию gpt-4)

    Returns:
        Количество токенов

    Пример:
        tokens = count_tokens("Hello, world!")
        print(f"Tokens: {tokens}")
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback на cl100k_base
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def estimate_cost(
    prompt: str,
    completion_tokens: int = 100,
    model: str = "gpt-4-turbo-preview"
) -> float:
    """
    Оценка стоимости запроса

    Args:
        prompt: Промпт
        completion_tokens: Ожидаемое количество токенов в ответе
        model: Модель

    Returns:
        Оценка стоимости в USD

    Пример:
        cost = estimate_cost("Summarize this...", completion_tokens=200)
        print(f"Estimated cost: ${cost:.4f}")
    """
    from .openai_client import OPENAI_PRICING
    from .anthropic_client import ANTHROPIC_PRICING

    prompt_tokens = count_tokens(prompt, model)

    # Определяем pricing
    if "gpt" in model.lower():
        pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4"])
    elif "claude" in model.lower():
        # Находим подходящий pricing
        for key in ANTHROPIC_PRICING.keys():
            if key in model:
                pricing = ANTHROPIC_PRICING[key]
                break
        else:
            pricing = ANTHROPIC_PRICING["claude-3-sonnet"]
    else:
        # Default на GPT-4
        pricing = OPENAI_PRICING["gpt-4"]

    prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1000) * pricing["completion"]

    return prompt_cost + completion_cost


def truncate_text(
    text: str,
    max_tokens: int = 2000,
    model: str = "gpt-4",
    suffix: str = "..."
) -> str:
    """
    Обрезать текст до max_tokens

    Args:
        text: Текст для обрезки
        max_tokens: Максимум токенов
        model: Модель для tokenizer
        suffix: Суффикс для добавления в конец

    Returns:
        Обрезанный текст

    Пример:
        text = "Very long text..."
        truncated = truncate_text(text, max_tokens=100)
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Обрезаем с учетом suffix
    suffix_tokens = encoding.encode(suffix)
    truncated_tokens = tokens[:max_tokens - len(suffix_tokens)]

    return encoding.decode(truncated_tokens) + suffix


def batch_generate(
    llm: LLMClient,
    prompts: list[str],
    **kwargs
) -> list[str]:
    """
    Batch generation для списка промптов

    Args:
        llm: LLM клиент
        prompts: Список промптов
        **kwargs: Параметры для generate()

    Returns:
        Список ответов

    Пример:
        llm = get_default_llm()
        results = batch_generate(llm, ["What is 2+2?", "What is 3+3?"])
    """
    results = []
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"Processing batch item {i}/{len(prompts)}")
        response = llm.generate(prompt, **kwargs)
        results.append(response.text)

    return results


def create_llm_from_name(provider: str, **kwargs) -> LLMClient:
    """
    Создать LLM клиент по имени провайдера

    Args:
        provider: Название провайдера ("openai" или "anthropic")
        **kwargs: Параметры для клиента

    Returns:
        LLMClient

    Пример:
        llm = create_llm_from_name("openai", model="gpt-3.5-turbo")
    """
    if provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
