"""
Базовые классы для LLM клиентов
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMResponse:
    """
    Унифицированный ответ от LLM

    Атрибуты:
        text: Сгенерированный текст
        model: Название использованной модели
        prompt_tokens: Количество токенов в промпте
        completion_tokens: Количество токенов в ответе
        total_tokens: Общее количество токенов
        cost: Стоимость запроса в USD
        latency: Время выполнения в секундах
        metadata: Дополнительные метаданные
    """
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


class LLMClient(ABC):
    """
    Абстрактный базовый класс для LLM клиентов

    Все LLM провайдеры должны наследовать этот класс и
    реализовать абстрактные методы.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Инициализация LLM клиента

        Args:
            model: Название модели
            temperature: Температура генерации (0.0-2.0)
            max_tokens: Максимальное количество токенов в ответе
            api_key: API ключ (если None, берётся из конфига)
            **kwargs: Дополнительные параметры
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.kwargs = kwargs

        # Статистика
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Генерация текста на основе промпта

        Args:
            prompt: Входной промпт
            temperature: Температура (переопределяет default)
            max_tokens: Максимум токенов (переопределяет default)
            **kwargs: Дополнительные параметры для API

        Returns:
            LLMResponse с результатом генерации
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Подсчет количества токенов в тексте

        Args:
            text: Текст для подсчета

        Returns:
            Количество токенов
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику использования

        Returns:
            Словарь со статистикой
        """
        return {
            "model": self.model,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "avg_tokens_per_request": (
                self.total_tokens / self.total_requests
                if self.total_requests > 0
                else 0
            ),
            "avg_cost_per_request": (
                self.total_cost / self.total_requests
                if self.total_requests > 0
                else 0
            ),
        }

    def reset_stats(self):
        """Сброс статистики"""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def _update_stats(self, response: LLMResponse):
        """
        Обновление статистики после запроса

        Args:
            response: Ответ от LLM
        """
        self.total_requests += 1
        self.total_tokens += response.total_tokens
        self.total_cost += response.cost

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
