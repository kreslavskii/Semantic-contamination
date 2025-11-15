"""
PairRM Ranker для парного ранжирования текстов

Использует llm-blender/PairRM для оценки качества пар текстов.
PairRM - это SOTA модель для pairwise ranking, обученная на данных сравнения
различных LLM выходов.

Модель работает как классификатор: даёт score для каждого из двух текстов,
и более высокий score указывает на лучший текст.
"""
import logging
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import warnings

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """
    Результат сравнения двух текстов

    Атрибуты:
        winner: 'A' если первый текст лучше, 'B' если второй, 'tie' если паритет
        score_a: Score первого текста
        score_b: Score второго текста
        confidence: Уверенность в решении (0-1)
        method: Метод оценки ('pairrm', 'heuristic', 'llm')
    """
    winner: str  # 'A', 'B', or 'tie'
    score_a: float
    score_b: float
    confidence: float
    method: str = "pairrm"

    def __str__(self) -> str:
        return f"Winner: {self.winner} (A={self.score_a:.3f}, B={self.score_b:.3f}, conf={self.confidence:.3f})"


class PairRMRanker:
    """
    PairRM-based ranker для парного ранжирования

    Пример использования:
        ranker = PairRMRanker()

        result = ranker.compare(
            text_a="Paris is the capital of France.",
            text_b="The capital of France is Paris, a major European city.",
            instruction="Which text is more informative?"
        )

        print(f"Winner: {result.winner}")  # 'B'
        print(f"Confidence: {result.confidence:.2f}")  # 0.85
    """

    MODEL_NAME = "llm-blender/PairRM"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_pairrm: Optional[bool] = None,
        batch_size: int = 1
    ):
        """
        Инициализация PairRM ranker

        Args:
            model_name: Название модели на HuggingFace (по умолчанию llm-blender/PairRM)
            device: Устройство ('cuda', 'cpu', или None для автоопределения)
            use_pairrm: Использовать ли PairRM (если None, берется из settings)
            batch_size: Размер батча для обработки
        """
        self.model_name = model_name or self.MODEL_NAME
        self.batch_size = batch_size
        self.use_pairrm = use_pairrm if use_pairrm is not None else settings.USE_PAIRRM

        # Проверка зависимостей
        if not HAS_TRANSFORMERS:
            logger.warning(
                "transformers или torch не установлены. "
                "PairRM будет недоступен. "
                "Установите: pip install transformers torch"
            )
            self.use_pairrm = False

        # Определение устройства
        if device is None:
            if torch and torch.cuda.is_available():
                self.device = "cuda"
                logger.info("CUDA доступна, используем GPU")
            else:
                self.device = "cpu"
                logger.info("CUDA недоступна, используем CPU")
        else:
            self.device = device

        # Загрузка модели
        self.model = None
        self.tokenizer = None

        if self.use_pairrm and HAS_TRANSFORMERS:
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"Ошибка загрузки PairRM: {e}")
                logger.warning("Переключаемся на heuristic fallback")
                self.use_pairrm = False

        # Статистика
        self.stats = {
            "comparisons": 0,
            "a_wins": 0,
            "b_wins": 0,
            "ties": 0,
            "errors": 0
        }

    def _load_model(self):
        """Загрузка PairRM модели и tokenizer"""
        logger.info(f"Загрузка PairRM модели: {self.model_name}")

        try:
            # Загружаем tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Загружаем модель
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            # Переводим в eval режим
            self.model.eval()

            logger.info(f"PairRM модель успешно загружена на {self.device}")

        except Exception as e:
            logger.error(f"Не удалось загрузить PairRM модель: {e}")
            raise

    def compare(
        self,
        text_a: str,
        text_b: str,
        instruction: str = "",
        tie_threshold: float = 0.1
    ) -> ComparisonResult:
        """
        Сравнение двух текстов

        Args:
            text_a: Первый текст
            text_b: Второй текст
            instruction: Контекст/вопрос/инструкция для сравнения
            tie_threshold: Порог для определения паритета (если разница score < threshold)

        Returns:
            ComparisonResult с победителем и scores

        Raises:
            ValueError: Если тексты пустые
        """
        if not text_a or not text_b:
            raise ValueError("Тексты не должны быть пустыми")

        self.stats["comparisons"] += 1

        # Режим PairRM
        if self.use_pairrm and self.model and self.tokenizer:
            try:
                return self._compare_with_pairrm(text_a, text_b, instruction, tie_threshold)
            except Exception as e:
                logger.error(f"Ошибка в PairRM сравнении: {e}")
                self.stats["errors"] += 1
                logger.warning("Переключаемся на heuristic fallback")

        # Fallback на эвристику
        return self._compare_with_heuristic(text_a, text_b, instruction)

    def _compare_with_pairrm(
        self,
        text_a: str,
        text_b: str,
        instruction: str,
        tie_threshold: float
    ) -> ComparisonResult:
        """
        Сравнение с использованием PairRM модели

        PairRM принимает на вход:
        - instruction (вопрос/контекст)
        - candidates (список текстов для сравнения)

        И возвращает scores для каждого кандидата.
        """
        # Подготовка входа для PairRM
        # Формат: [instruction, candidate_1] и [instruction, candidate_2]
        inputs_a = self.tokenizer(
            [instruction],
            [text_a],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        inputs_b = self.tokenizer(
            [instruction],
            [text_b],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Инференс
        with torch.no_grad():
            # PairRM возвращает logits для каждого кандидата
            outputs_a = self.model(**inputs_a)
            outputs_b = self.model(**inputs_b)

            # Извлекаем scores (logits[:, 1] - score для "лучше")
            score_a = outputs_a.logits[0, 1].item()
            score_b = outputs_b.logits[0, 1].item()

        # Определяем победителя
        score_diff = abs(score_a - score_b)

        if score_diff < tie_threshold:
            winner = "tie"
            self.stats["ties"] += 1
        elif score_a > score_b:
            winner = "A"
            self.stats["a_wins"] += 1
        else:
            winner = "B"
            self.stats["b_wins"] += 1

        # Confidence = нормализованная разница scores
        # Используем sigmoid для нормализации
        confidence = self._calculate_confidence(score_diff)

        return ComparisonResult(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            confidence=confidence,
            method="pairrm"
        )

    def _calculate_confidence(self, score_diff: float) -> float:
        """
        Вычисление confidence из разницы scores

        Args:
            score_diff: Абсолютная разница между scores

        Returns:
            Confidence (0-1)
        """
        # Используем сигмоиду для отображения в [0, 1]
        # score_diff обычно в диапазоне [0, 10+]
        # Нормализуем так, чтобы diff=1 давал ~0.73 confidence
        if torch:
            return torch.sigmoid(torch.tensor(score_diff)).item()
        else:
            # Fallback если torch недоступен
            import math
            return 1 / (1 + math.exp(-score_diff))

    def _compare_with_heuristic(
        self,
        text_a: str,
        text_b: str,
        instruction: str
    ) -> ComparisonResult:
        """
        Fallback сравнение с использованием простых эвристик

        Эвристики:
        1. Длина текста (более подробный обычно лучше)
        2. Наличие числовых данных
        3. Структурированность (наличие списков, пунктуации)
        """
        score_a = 0.0
        score_b = 0.0

        # 1. Оценка по длине (более детальный ответ лучше, но не слишком длинный)
        len_a = len(text_a.split())
        len_b = len(text_b.split())

        ideal_length = 50  # Оптимальная длина
        score_a += 1 - abs(len_a - ideal_length) / ideal_length
        score_b += 1 - abs(len_b - ideal_length) / ideal_length

        # 2. Наличие конкретных данных (цифры, даты)
        import re
        numbers_a = len(re.findall(r'\d+', text_a))
        numbers_b = len(re.findall(r'\d+', text_b))

        score_a += min(numbers_a / 3, 1.0)  # Cap at 3 numbers
        score_b += min(numbers_b / 3, 1.0)

        # 3. Структурированность
        # Проверяем наличие пунктуации, организации текста
        structure_markers = [',', '.', ';', ':', '—', '–']
        structure_a = sum(text_a.count(m) for m in structure_markers)
        structure_b = sum(text_b.count(m) for m in structure_markers)

        score_a += min(structure_a / 5, 1.0)
        score_b += min(structure_b / 5, 1.0)

        # Нормализация
        score_a = score_a / 3
        score_b = score_b / 3

        # Определение победителя
        if abs(score_a - score_b) < 0.1:
            winner = "tie"
            self.stats["ties"] += 1
        elif score_a > score_b:
            winner = "A"
            self.stats["a_wins"] += 1
        else:
            winner = "B"
            self.stats["b_wins"] += 1

        confidence = abs(score_a - score_b)

        return ComparisonResult(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            confidence=confidence,
            method="heuristic"
        )

    def compare_multiple(
        self,
        pairs: List[Tuple[str, str, str]]
    ) -> List[ComparisonResult]:
        """
        Сравнение нескольких пар текстов

        Args:
            pairs: Список кортежей (text_a, text_b, instruction)

        Returns:
            Список ComparisonResult
        """
        results = []

        for text_a, text_b, instruction in pairs:
            result = self.compare(text_a, text_b, instruction)
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Получить статистику использования"""
        stats = self.stats.copy()

        total = stats["comparisons"]
        if total > 0:
            stats["a_win_rate"] = (stats["a_wins"] / total) * 100
            stats["b_win_rate"] = (stats["b_wins"] / total) * 100
            stats["tie_rate"] = (stats["ties"] / total) * 100
            stats["error_rate"] = (stats["errors"] / total) * 100

        stats["mode"] = "PairRM" if self.use_pairrm else "Heuristic"
        stats["device"] = self.device if self.use_pairrm else "N/A"

        return stats

    def reset_stats(self):
        """Сброс статистики"""
        self.stats = {
            "comparisons": 0,
            "a_wins": 0,
            "b_wins": 0,
            "ties": 0,
            "errors": 0
        }


# Пример использования
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Создание ranker
    ranker = PairRMRanker()

    # Пример сравнения
    text_a = "Париж — столица Франции."
    text_b = "Столица Франции — Париж, крупнейший город страны с населением более 2 миллионов человек."
    instruction = "Какой текст более информативен о столице Франции?"

    result = ranker.compare(text_a, text_b, instruction)

    print(f"\nРезультат сравнения:")
    print(f"Текст A: {text_a}")
    print(f"Текст B: {text_b}")
    print(f"Вопрос: {instruction}")
    print(f"\n{result}")
    print(f"\nСтатистика:")
    for key, value in ranker.get_stats().items():
        print(f"  {key}: {value}")
