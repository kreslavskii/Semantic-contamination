"""
Cost tracking для LLM API вызовов
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Запись о стоимости одного запроса"""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    task: str = "unknown"

    def to_dict(self) -> Dict:
        """Конвертация в словарь для JSON"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'CostEntry':
        """Создание из словаря"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class CostTracker:
    """
    Отслеживание расходов на LLM API

    Пример использования:
        tracker = CostTracker()
        tracker.add_request("gpt-4", 100, 50, 0.015, "extraction")
        print(f"Total cost: ${tracker.get_total_cost():.4f}")
        tracker.save()  # Сохранить в файл
    """

    def __init__(self, save_path: Optional[Path] = None):
        """
        Инициализация tracker

        Args:
            save_path: Путь к файлу для сохранения истории
        """
        self.save_path = save_path or (settings.output_path / "cost_history.json")
        self.entries: List[CostEntry] = []

        # Загрузить существующую историю если есть
        self.load()

    def add_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        task: str = "unknown"
    ):
        """
        Добавить запись о запросе

        Args:
            model: Название модели
            prompt_tokens: Токены промпта
            completion_tokens: Токены ответа
            cost_usd: Стоимость в USD
            task: Название задачи (например, "extraction", "verification")
        """
        entry = CostEntry(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            task=task
        )
        self.entries.append(entry)

        # Проверка лимита
        total_cost = self.get_total_cost()
        if total_cost > settings.MAX_COST_PER_RUN:
            logger.warning(
                f"Cost limit exceeded: ${total_cost:.4f} > "
                f"${settings.MAX_COST_PER_RUN:.2f}"
            )

    def get_total_cost(self) -> float:
        """Получить общую стоимость всех запросов"""
        return sum(entry.cost_usd for entry in self.entries)

    def get_total_tokens(self) -> int:
        """Получить общее количество токенов"""
        return sum(entry.total_tokens for entry in self.entries)

    def get_stats_by_model(self) -> Dict[str, Dict]:
        """
        Статистика по моделям

        Returns:
            Словарь {model: {requests, tokens, cost}}
        """
        stats = {}
        for entry in self.entries:
            if entry.model not in stats:
                stats[entry.model] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0
                }

            stats[entry.model]["requests"] += 1
            stats[entry.model]["tokens"] += entry.total_tokens
            stats[entry.model]["cost"] += entry.cost_usd

        return stats

    def get_stats_by_task(self) -> Dict[str, Dict]:
        """
        Статистика по задачам

        Returns:
            Словарь {task: {requests, tokens, cost}}
        """
        stats = {}
        for entry in self.entries:
            if entry.task not in stats:
                stats[entry.task] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0
                }

            stats[entry.task]["requests"] += 1
            stats[entry.task]["tokens"] += entry.total_tokens
            stats[entry.task]["cost"] += entry.cost_usd

        return stats

    def save(self):
        """Сохранить историю в файл"""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "saved_at": datetime.now().isoformat(),
                "total_cost": self.get_total_cost(),
                "total_tokens": self.get_total_tokens(),
                "entries": [entry.to_dict() for entry in self.entries]
            }

            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Cost history saved to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save cost history: {e}")

    def load(self):
        """Загрузить историю из файла"""
        if not self.save_path.exists():
            return

        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.entries = [
                CostEntry.from_dict(entry_dict)
                for entry_dict in data.get("entries", [])
            ]

            logger.info(
                f"Loaded {len(self.entries)} cost entries from {self.save_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load cost history: {e}")

    def reset(self):
        """Сброс всех записей"""
        self.entries = []
        logger.info("Cost tracker reset")

    def print_summary(self):
        """Вывод сводки расходов"""
        print("=" * 60)
        print("LLM API Cost Summary")
        print("=" * 60)

        print(f"\nTotal Requests: {len(self.entries)}")
        print(f"Total Tokens: {self.get_total_tokens():,}")
        print(f"Total Cost: ${self.get_total_cost():.4f}")

        print("\n--- By Model ---")
        for model, stats in self.get_stats_by_model().items():
            print(f"{model}:")
            print(f"  Requests: {stats['requests']}")
            print(f"  Tokens: {stats['tokens']:,}")
            print(f"  Cost: ${stats['cost']:.4f}")

        print("\n--- By Task ---")
        for task, stats in self.get_stats_by_task().items():
            print(f"{task}:")
            print(f"  Requests: {stats['requests']}")
            print(f"  Tokens: {stats['tokens']:,}")
            print(f"  Cost: ${stats['cost']:.4f}")

        print("=" * 60)


# Глобальный tracker (singleton)
_global_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Получить глобальный cost tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker
