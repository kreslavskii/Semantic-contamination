"""
Централизованная конфигурация для проекта Semantic Contamination

Использует pydantic для валидации настроек из переменных окружения.
Читает из .env файла если он существует.

Пример использования:
    from config import settings

    print(settings.DEFAULT_MODEL)
    print(settings.USE_PAIRRM)
"""
import os
from typing import Optional, Literal
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, validator


# Определяем базовую директорию проекта
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Настройки проекта с валидацией

    Все значения читаются из переменных окружения или .env файла.
    Если переменная не задана, используется значение по умолчанию.
    """

    # ============================================
    # LLM API Keys
    # ============================================
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key для GPT-4/3.5"
    )

    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key для Claude"
    )

    # ============================================
    # Web Search API
    # ============================================
    SERPAPI_API_KEY: Optional[str] = Field(
        default=None,
        description="SerpAPI key для веб-поиска"
    )

    # ============================================
    # Model Settings
    # ============================================
    DEFAULT_LLM_PROVIDER: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Провайдер LLM по умолчанию"
    )

    DEFAULT_MODEL: str = Field(
        default="gpt-4-turbo-preview",
        description="Модель для использования"
    )

    TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Температура для generation"
    )

    MAX_TOKENS: int = Field(
        default=2000,
        gt=0,
        description="Максимальное количество токенов в ответе"
    )

    # ============================================
    # Agent Settings
    # ============================================
    USE_PAIRRM: bool = Field(
        default=True,
        description="Использовать PairRM для JudgeAgent"
    )

    USE_SELFCHECK: bool = Field(
        default=True,
        description="Использовать SelfCheckGPT для ExtractorAgent"
    )

    SELFCHECK_SAMPLES: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Количество samples для SelfCheck"
    )

    JUDGE_DOUBLE_CHECK: bool = Field(
        default=True,
        description="Double-check для JudgeAgent"
    )

    # ============================================
    # Cost & Rate Limits
    # ============================================
    MAX_COST_PER_RUN: float = Field(
        default=10.0,
        ge=0.0,
        description="Максимальная стоимость одного запуска (USD)"
    )

    MAX_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        gt=0,
        description="Максимум API вызовов в минуту"
    )

    API_TIMEOUT: int = Field(
        default=30,
        gt=0,
        description="Таймаут для API запросов (секунды)"
    )

    # ============================================
    # Logging
    # ============================================
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Уровень логирования"
    )

    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Путь к файлу логов (None = консоль)"
    )

    # ============================================
    # Paths
    # ============================================
    INPUT_DIR: str = Field(
        default="data",
        description="Директория с входными документами"
    )

    OUTPUT_DIR: str = Field(
        default="output",
        description="Директория для результатов"
    )

    PROMPTS_DIR: str = Field(
        default="prompts",
        description="Директория с промптами"
    )

    # ============================================
    # Performance
    # ============================================
    USE_GPU: bool = Field(
        default=True,
        description="Использовать GPU для локальных моделей"
    )

    NUM_WORKERS: int = Field(
        default=4,
        ge=1,
        description="Количество worker threads"
    )

    BATCH_SIZE: int = Field(
        default=10,
        ge=1,
        description="Размер батча для batch processing"
    )

    # ============================================
    # Cache Settings
    # ============================================
    ENABLE_CACHE: bool = Field(
        default=True,
        description="Кэшировать LLM ответы"
    )

    CACHE_TTL: int = Field(
        default=3600,
        ge=0,
        description="Время жизни кэша (секунды)"
    )

    CACHE_DIR: str = Field(
        default=".cache",
        description="Директория для кэша"
    )

    # ============================================
    # Development
    # ============================================
    DEBUG: bool = Field(
        default=False,
        description="Debug mode"
    )

    DRY_RUN: bool = Field(
        default=False,
        description="Dry run (моки вместо API)"
    )

    # ============================================
    # Computed Properties
    # ============================================

    @property
    def input_path(self) -> Path:
        """Абсолютный путь к INPUT_DIR"""
        return PROJECT_ROOT / self.INPUT_DIR

    @property
    def output_path(self) -> Path:
        """Абсолютный путь к OUTPUT_DIR"""
        return PROJECT_ROOT / self.OUTPUT_DIR

    @property
    def prompts_path(self) -> Path:
        """Абсолютный путь к PROMPTS_DIR"""
        return PROJECT_ROOT / self.PROMPTS_DIR

    @property
    def cache_path(self) -> Path:
        """Абсолютный путь к CACHE_DIR"""
        return PROJECT_ROOT / self.CACHE_DIR

    @property
    def has_openai_key(self) -> bool:
        """Проверка наличия OpenAI API key"""
        return self.OPENAI_API_KEY is not None and len(self.OPENAI_API_KEY) > 0

    @property
    def has_anthropic_key(self) -> bool:
        """Проверка наличия Anthropic API key"""
        return self.ANTHROPIC_API_KEY is not None and len(self.ANTHROPIC_API_KEY) > 0

    @property
    def has_serpapi_key(self) -> bool:
        """Проверка наличия SerpAPI key"""
        return self.SERPAPI_API_KEY is not None and len(self.SERPAPI_API_KEY) > 0

    @property
    def can_use_llm(self) -> bool:
        """Проверка возможности использовать LLM"""
        if self.DEFAULT_LLM_PROVIDER == "openai":
            return self.has_openai_key
        elif self.DEFAULT_LLM_PROVIDER == "anthropic":
            return self.has_anthropic_key
        return False

    # ============================================
    # Validators
    # ============================================

    @validator("DEFAULT_MODEL")
    def validate_model(cls, v, values):
        """Валидация совместимости модели с провайдером"""
        provider = values.get("DEFAULT_LLM_PROVIDER", "openai")

        openai_models = ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"]
        anthropic_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

        if provider == "openai" and not any(m in v for m in openai_models):
            raise ValueError(f"Model {v} не совместима с провайдером {provider}")

        if provider == "anthropic" and not any(m in v for m in anthropic_models):
            raise ValueError(f"Model {v} не совместима с провайдером {provider}")

        return v

    def ensure_directories(self):
        """Создание необходимых директорий"""
        self.input_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
        self.prompts_path.mkdir(exist_ok=True)
        if self.ENABLE_CACHE:
            self.cache_path.mkdir(exist_ok=True)

    class Config:
        # Читать из .env файла если он существует
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Игнорировать extra поля
        extra = "ignore"
        # Case sensitive переменные
        case_sensitive = True


# Создаём глобальный экземпляр настроек
settings = Settings()


# Создаём необходимые директории при импорте
settings.ensure_directories()


# ============================================
# Utility Functions
# ============================================

def get_api_key(provider: str) -> Optional[str]:
    """
    Получить API ключ для провайдера

    Args:
        provider: Название провайдера ('openai', 'anthropic', 'serpapi')

    Returns:
        API ключ или None
    """
    if provider == "openai":
        return settings.OPENAI_API_KEY
    elif provider == "anthropic":
        return settings.ANTHROPIC_API_KEY
    elif provider == "serpapi":
        return settings.SERPAPI_API_KEY
    return None


def validate_api_keys():
    """
    Проверка наличия необходимых API ключей

    Raises:
        ValueError: Если отсутствуют критичные ключи
    """
    if not settings.can_use_llm:
        raise ValueError(
            f"API ключ для {settings.DEFAULT_LLM_PROVIDER} не найден. "
            f"Установите {settings.DEFAULT_LLM_PROVIDER.upper()}_API_KEY в .env файле"
        )


def print_config():
    """Вывод текущей конфигурации (без секретов)"""
    print("=" * 50)
    print("Semantic Contamination - Configuration")
    print("=" * 50)
    print(f"LLM Provider: {settings.DEFAULT_LLM_PROVIDER}")
    print(f"Model: {settings.DEFAULT_MODEL}")
    print(f"Temperature: {settings.TEMPERATURE}")
    print(f"Max Tokens: {settings.MAX_TOKENS}")
    print(f"")
    print(f"OpenAI API: {'✓' if settings.has_openai_key else '✗'}")
    print(f"Anthropic API: {'✓' if settings.has_anthropic_key else '✗'}")
    print(f"SerpAPI: {'✓' if settings.has_serpapi_key else '✗'}")
    print(f"")
    print(f"Use PairRM: {settings.USE_PAIRRM}")
    print(f"Use SelfCheck: {settings.USE_SELFCHECK}")
    print(f"SelfCheck Samples: {settings.SELFCHECK_SAMPLES}")
    print(f"")
    print(f"Input Dir: {settings.input_path}")
    print(f"Output Dir: {settings.output_path}")
    print(f"Prompts Dir: {settings.prompts_path}")
    print(f"")
    print(f"GPU Enabled: {settings.USE_GPU}")
    print(f"Cache Enabled: {settings.ENABLE_CACHE}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Dry Run: {settings.DRY_RUN}")
    print("=" * 50)


# Пример использования
if __name__ == "__main__":
    print_config()

    # Проверка API ключей
    try:
        validate_api_keys()
        print("\n✓ Все необходимые API ключи настроены")
    except ValueError as e:
        print(f"\n✗ Ошибка конфигурации: {e}")
