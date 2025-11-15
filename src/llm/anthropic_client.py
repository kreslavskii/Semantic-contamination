"""
Anthropic (Claude) LLM Client
"""
import time
import logging
from typing import Optional
from anthropic import Anthropic, RateLimitError, APITimeoutError, APIConnectionError

from .base import LLMClient, LLMResponse
from ..config import settings

logger = logging.getLogger(__name__)


# Стоимость моделей Anthropic (USD за 1000 токенов)
# Актуально на 2025-11
ANTHROPIC_PRICING = {
    "claude-3-opus": {
        "prompt": 0.015,
        "completion": 0.075,
    },
    "claude-3-sonnet": {
        "prompt": 0.003,
        "completion": 0.015,
    },
    "claude-3-haiku": {
        "prompt": 0.00025,
        "completion": 0.00125,
    },
}


class AnthropicClient(LLMClient):
    """
    Anthropic (Claude) API клиент

    Пример использования:
        client = AnthropicClient(model="claude-3-sonnet-20240229")
        response = client.generate("What is the capital of France?")
        print(response.text)
    """

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Инициализация Anthropic клиента

        Args:
            model: Название модели Claude
            temperature: Температура генерации
            max_tokens: Максимум токенов в ответе
            api_key: API ключ (если None, берётся из settings)
            max_retries: Максимальное количество попыток при ошибке
            retry_delay: Задержка между попытками
            **kwargs: Дополнительные параметры
        """
        super().__init__(model, temperature, max_tokens, api_key, **kwargs)

        self.api_key = api_key or settings.ANTHROPIC_API_KEY

        if not self.api_key:
            raise ValueError(
                "Anthropic API key не найден. Установите ANTHROPIC_API_KEY "
                "в .env файле или передайте api_key параметр"
            )

        # Инициализация клиента
        self.client = Anthropic(api_key=self.api_key)

        # Retry settings
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"AnthropicClient инициализирован: model={model}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Генерация текста через Anthropic API

        Args:
            prompt: Входной промпт
            temperature: Температура
            max_tokens: Максимум токенов
            **kwargs: Дополнительные параметры

        Returns:
            LLMResponse с результатом
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if settings.DRY_RUN:
            return self._mock_response(prompt)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tok,
                    temperature=temp,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **{**self.kwargs, **kwargs}
                )

                latency = time.time() - start_time

                # Извлечение данных
                text = response.content[0].text
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
                total_tokens = prompt_tokens + completion_tokens

                # Стоимость
                cost = self._calculate_cost(prompt_tokens, completion_tokens)

                llm_response = LLMResponse(
                    text=text,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                    latency=latency,
                    metadata={
                        "stop_reason": response.stop_reason,
                        "attempt": attempt + 1,
                    }
                )

                self._update_stats(llm_response)

                logger.info(
                    f"Anthropic request successful: "
                    f"tokens={total_tokens}, cost=${cost:.4f}, "
                    f"latency={latency:.2f}s"
                )

                return llm_response

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                logger.warning(
                    f"API error (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

            except Exception as e:
                logger.error(f"Unexpected error in Anthropic request: {e}")
                raise

        logger.error(f"All retry attempts failed: {last_error}")
        raise last_error

    def count_tokens(self, text: str) -> int:
        """
        Подсчет токенов (приблизительный)

        Anthropic использует собственный tokenizer,
        но мы можем аппроксимировать как ~4 символа на токен
        """
        return len(text) // 4

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Расчет стоимости запроса"""
        pricing = None
        for model_key, model_pricing in ANTHROPIC_PRICING.items():
            if model_key in self.model:
                pricing = model_pricing
                break

        if not pricing:
            logger.warning(
                f"Pricing для модели {self.model} не найден, "
                f"используем pricing для claude-3-sonnet"
            )
            pricing = ANTHROPIC_PRICING["claude-3-sonnet"]

        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    def _mock_response(self, prompt: str) -> LLMResponse:
        """Mock ответ для DRY_RUN"""
        mock_text = "[DRY RUN] This is a mock Claude response"
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(mock_text)

        return LLMResponse(
            text=mock_text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=0.0,
            latency=0.1,
            metadata={"dry_run": True}
        )
