"""
OpenAI LLM Client с retry logic и cost tracking
"""
import time
import logging
from typing import Optional
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
import tiktoken

from .base import LLMClient, LLMResponse
from ..config import settings

logger = logging.getLogger(__name__)


# Стоимость моделей OpenAI (USD за 1000 токенов)
# Актуально на 2025-11
OPENAI_PRICING = {
    "gpt-4-turbo-preview": {
        "prompt": 0.01,      # $0.01 per 1K prompt tokens
        "completion": 0.03,  # $0.03 per 1K completion tokens
    },
    "gpt-4": {
        "prompt": 0.03,
        "completion": 0.06,
    },
    "gpt-3.5-turbo": {
        "prompt": 0.0005,
        "completion": 0.0015,
    },
}


class OpenAIClient(LLMClient):
    """
    OpenAI API клиент с поддержкой retry logic и cost tracking

    Пример использования:
        client = OpenAIClient(model="gpt-4-turbo-preview")
        response = client.generate("What is 2+2?")
        print(response.text)
        print(f"Cost: ${response.cost:.4f}")
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Инициализация OpenAI клиента

        Args:
            model: Название модели OpenAI
            temperature: Температура генерации
            max_tokens: Максимум токенов в ответе
            api_key: API ключ (если None, берётся из settings)
            max_retries: Максимальное количество попыток при ошибке
            retry_delay: Задержка между попытками (секунды)
            **kwargs: Дополнительные параметры для OpenAI API
        """
        super().__init__(model, temperature, max_tokens, api_key, **kwargs)

        # API ключ из параметра или settings
        self.api_key = api_key or settings.OPENAI_API_KEY

        if not self.api_key:
            raise ValueError(
                "OpenAI API key не найден. Установите OPENAI_API_KEY "
                "в .env файле или передайте api_key параметр"
            )

        # Инициализация OpenAI клиента
        self.client = OpenAI(api_key=self.api_key)

        # Retry settings
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Tokenizer для подсчета токенов
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback на cl100k_base для новых моделей
            logger.warning(
                f"Tokenizer для модели {model} не найден, "
                f"используем cl100k_base"
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"OpenAIClient инициализирован: model={model}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Генерация текста через OpenAI API с retry logic

        Args:
            prompt: Входной промпт
            temperature: Температура (переопределяет default)
            max_tokens: Максимум токенов (переопределяет default)
            **kwargs: Дополнительные параметры для API

        Returns:
            LLMResponse с результатом

        Raises:
            Exception: После исчерпания всех попыток retry
        """
        # Параметры с fallback на defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Сухой прогон для тестирования
        if settings.DRY_RUN:
            return self._mock_response(prompt)

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # API вызов
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=max_tok,
                    **{**self.kwargs, **kwargs}
                )

                latency = time.time() - start_time

                # Извлечение данных
                text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # Подсчет стоимости
                cost = self._calculate_cost(prompt_tokens, completion_tokens)

                # Формирование ответа
                llm_response = LLMResponse(
                    text=text,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                    latency=latency,
                    metadata={
                        "finish_reason": response.choices[0].finish_reason,
                        "attempt": attempt + 1,
                    }
                )

                # Обновление статистики
                self._update_stats(llm_response)

                logger.info(
                    f"OpenAI request successful: "
                    f"tokens={total_tokens}, cost=${cost:.4f}, "
                    f"latency={latency:.2f}s"
                )

                return llm_response

            except RateLimitError as e:
                last_error = e
                logger.warning(
                    f"Rate limit exceeded (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {self.retry_delay}s..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

            except (APITimeoutError, APIConnectionError) as e:
                last_error = e
                logger.warning(
                    f"API connection error (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {self.retry_delay}s..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error in OpenAI request: {e}")
                raise

        # Если все попытки исчерпаны
        logger.error(f"All retry attempts failed: {last_error}")
        raise last_error

    def count_tokens(self, text: str) -> int:
        """
        Подсчет токенов в тексте

        Args:
            text: Текст для подсчета

        Returns:
            Количество токенов
        """
        return len(self.tokenizer.encode(text))

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Расчет стоимости запроса

        Args:
            prompt_tokens: Количество токенов в промпте
            completion_tokens: Количество токенов в ответе

        Returns:
            Стоимость в USD
        """
        # Находим pricing для модели (поддержка partial match)
        pricing = None
        for model_key, model_pricing in OPENAI_PRICING.items():
            if model_key in self.model:
                pricing = model_pricing
                break

        if not pricing:
            logger.warning(
                f"Pricing для модели {self.model} не найден, "
                f"используем pricing для gpt-4"
            )
            pricing = OPENAI_PRICING["gpt-4"]

        # Расчет стоимости
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    def _mock_response(self, prompt: str) -> LLMResponse:
        """
        Mock ответ для DRY_RUN режима

        Args:
            prompt: Промпт (для подсчета токенов)

        Returns:
            Mock LLMResponse
        """
        mock_text = "[DRY RUN] This is a mock response"
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
